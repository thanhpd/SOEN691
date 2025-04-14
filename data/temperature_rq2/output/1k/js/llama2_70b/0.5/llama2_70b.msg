Update trackCore.js: Remove unused property and add default color for new tracks
Fix ConnectionManager._processNext to only run when there is no current task.
ResponsivePopover: Set begin and end button types to Transparent.
Change loop mode of hand controls from PingPong to LoopRepeat
Added constants stub for fastfile
Fixed test for report with end date before start date
Added istanbul ignore flag for untestable code in _gpfExit function.
Added logging for error handling in destroyRole() in spec/helpers/generic-role.js
Updated third-party.js to strip query strings from module URLs and use the correct file path for modules.
Added templateError and test1 tasks to Gruntfile.js
fix: correctly handle v-model with array children
Update dirtyForms.js to correctly detect IE8 and IE9 versions.
Replace cordova/commandProxy with cordova/windows8/commandProxy in AccelerometerProxy.js
Update getFiles function to handle Webpack 4 entrypoints
Added Errors global variable and updated throw statement in deriveAddresses method.
Update require path for ui-layout
Added variable _VALIDATIONS_REF to store validation messages and updated validationError function to use it.
Removed unused variable gen_keys
Added unique identifier for uploaded files
Replace 'return $(this).each()' with 'return this.each()' in jquery.peity.js
Replace iframe with iframe#editJob in templates.js
Update trace.js to use original callSite instead of callSite.slice(2)
Update animation type from 'slide' to 'fade' in RkPicker component.
Update ancestor declaration for gradient filter
Added keyDown function to ValidatedSelectComponent to trigger save action on enter key press and added observer for parentModel errors.
Update datatable.js to modify the selection of rows based on the search filter.
Added 'data-id' attribute to region elements in WaveSurfer.Region
Added program.option() and updated command registration
Allow IconButton to accept SvgIcon children
Fix for Quill's shiftRange function to correctly handle user-generated content.
Update selectors optimizer with new options
Added overwrites parameter to lines method in TextMetrics class

Added variable self to AppPermissions.prototype.checkPackageContentsExists for unit testing
Fixed type checking for data and options parameters in sendPushNotification method.
Added ability to search for agents by UUID or address, and removed hidden agents from search results.
Increased network update frequency and added code to handle movement commands.
Added flag to executeScript to disable path optimization
Export current running test in PlaybackState
Update exec.js to use webpack.js instead of webpack.prod.js

Replace 'addComponents' with 'addUtilities' in index.js
Removed physics engine initialization and updated event emitter setup.
Added db and password to queueConfig object in Bull constructor
Update options object with new defaults
Update bundleWebpack configuration to use {configPath: options.webpackConfigPath} and add crossorigin: false for HtmlScript followRelations.
Added Twitch IRC client with support for SSL connections and updated server and port settings.
Update RangeSlider.js to fix issue with _updateHandle and fireLiveChange instead of _fireChangeAndLiveChange
Added setCustomMarker function call in compile method of custom-marker.js
Icon: Update iconFamily condition for Ionicons
Update jest.config.js
Modified Server class to use 'done' callback instead of 'callback' in pool destroy method.
Added conditional height adjustment for slideshow widget.
Added support for SmartThings Smart Siren device
Update publish flow confirm button text
Added ancestor and children properties to RenderableContainer, and updated sort method to use z value when available
Update symbols for Windows platforms in cli-utils.js
Fixed issue with null values not being properly formatted in observable text input
Implemented frame consumption and fixed byte consumption for unsupported frame types.
Added environment variables to config.js
Added sudo to route command in getDefaultInterface function.
Fix Redback constructor to accept a redis client object or create one if not provided.
Updated editor plugin to disable/enable link button based on node name and added respondToNodeChange() function to link form.
Added support for transceiver requests and renegotiations in Peer class.
Added deprecation warning for 'validationObject' property in favor of 'validationModel' in FlexberryEditFormControllerMixin.
Update test specs and clean up preferencesKey
Added support for touch devices with multiple touch points
Added support for Chrome event handling
Update test_modrewrite.js to use grunt-contrib-connect-modrewrite instead of Authentication
Fixed error in uncss.js where processFile was not being called when there was only one file.
Added support for invoking generators on Chance instances.
Fixed lexer tokenization for paragraphs with trailing newlines.
Update streamCompare function to accept options or comparison function as last argument, and remove unnecessary 'typeof' checks for Promise.

Fixed syncer test to use correct API URL
Update Text component to use highlightString and highlightStyle props, and add new styles for highlighted and non-highlighted text.
Added exception for Ionic app clicks in gesture.js
 Fixes #12345: Filter out falsy values in getVnodes.
Added selector for getting connection queue.
Update WizardProgressNavigator.js to use getCurrentStep() instead of _activeStep
Updated admin/common.js to enhance dynamic tables by adding and removing rows.
Update typography theme Irving's base font size and line height
Replace __dirname with __filename in Generator.resolved
Add const for SIFTSCIENCE_URL and hasLoaded
Added before property to VerifyStream and updated gunzip and writable properties.
Update exclude pattern in karma-minified.conf.js
Update to use ES6 syntax for importing core.js
TextReplay: Simplified text rendering
Added cache whitelist checking in TritonAgent.js
Update error message format to include server error information
Added getLanguageId() method to NullLanguageMode class.
Added warning for missing packages in npm.js
Update dependencies for CorrelationService and DeleteProcessModelService
Update passport-vkontakte strategy.js to use _scope instead of options.scope for fetching user data
Fix error handling in sendMessageProto
Update Text styles with new color and letterSpacing variants
Updated version and uWebSockets version logging in details.js
Update statsporn.js with new regular expressions for command and help matches, and added a cooldown timer array
Added support for Error.captureStackTrace() in StatusCodeError constructor.
Added test for StyledComponent creation order
Added visit tracking and unique event id generation
Added console.group() and console.groupEnd() for debugging purposes in resolveRef().
Update 'u.withDefault' spec to use frozen objects
Updated conversation model to use 'lastMessage' instead of 'body' for the last message in a conversation.
Update utils.js to use HTTPS protocol for https requests
Update highlight.js to use hljs.highlight() with ignoreIllegals: true
Added functionality to handle dropped files
Fix for ProgressBar.prototype.tick.mock.calls.length not equaling modules.length in Bundler-test.js.
Update test.js to use waitFor instead of w
Update test for Windows Phone platform
Update cat module from require('__dirname + '/index.js') to require('./index')
Added Thai language support to designer controller
Added 'values' property to Statistics class and updated total property to use toFixed() method.
Update inject.js to improve performance by using Reflect.ownKeys instead of Object.keys, and ignoring symbol properties.
Update interact.js to version 1.0.2 with strict mode and export changes.
Update lint-bare-strings.js: Add '•' and '—' to whitelist
Added support for retryable reads with server selection
Replace arrayBufferToBuffer with Buffer.from
Increased number of offers generated by WebSocketTracker from 5 to 10.
Update rollup.config.js to use esm format instead of es
Reverse quoteVolume and baseVolume in gateio.js
Modified query-helper.js to exclude virtual fields and fields with $* suffix in filter conditions.
Added timestamp parameter to Spotify.getFeaturedPlaylists() method to get fresh data
Changes in bitstamp.js: updated timedelta calculation and added milliseconds function.
Added console logs for loading test modules and updated require statement for Ractive.

Added Lodash methods pick and isEmpty to logging server
Added two new benchmark fixtures and updated suite.js
Added rimraf for cleaning up temporary files in generate.spec.js
ErrorFormatter: added trimming of long lines
Added return statement in quit function

Added support for parsing key-value pairs with leading and trailing spaces
Added check for existing destruction before removing extent rectangle in OverviewMap control.
Renamed Jetpack Connect routes for site-type and site-topic
Removed unused dependency 'path' in createCacheKeyFunction.js
Added check for bid and ask prices in testTicker function for non-Coinmarketcap exchanges.
Add test for valid attribute 'isPrimary' and 'primary'.
Added support for transforming JavaScript files with comments and requires.
Fixed handling of successful responses in createAPIMiddleware
Fix redis test to use 'should respond without an error' instead of 'should respond without an' to make it more clear and consistent with other tests.
Fix for zookeeper.js: Update 'that.emit' to use 'err' instead of 'error' in listBrokers method
Added errno to runner globals
Renamed GitLibraryNode to SvnLibraryNode and updated inheritance
Added deployments for MockOracle and ETHOracle contracts.
 Fixes issue with incorrect positioning of cursor when clicking on a span element.
Update action creators to use object-assign instead of Object.assign
Update webview/browser/pre/main.js: Add missing \n in DOCTYPE declaration and remove unnecessary writes.
Added protection for inline editing
Fixed issue with traverse function not working correctly with frozen objects.
Fix for missing analytics attributes on elements
Update generate-choco-package.js to use yargs-parser instead of minimist
Added code to handle loading of URLs without a protocol
Removed unused exchange 'coss' and updated 'zb' exchange to use the correct require statement.
Modified processor.js to use `getKey()` instead of `getEntityAt(0)` for getting the key of a block.
Update MJMLColumnElement.js to use mjAttribute for padding and remove unnecessary padding properties.
Update camera.js to use hasOwnProperty() to check for existence of 'Scopes' property in GetScopesResponse before assigning it to variable.
Added peer statistics logging interval
Added function rinfo to construct remote information object and updated server initialization to use it.
 Fix typo in redux.js: removed unnecessary '| void' in type definition of Reducer
Added return statement in copyInfo function.
Update dependency versions to use tilde ranges for actual releases
Added check for existence of infoGrid element before returning it.
Updated i18n translation for en-us.js
 Fix: Remove unnecessary comma in StepGroup.js
Added support for multiple selections in highlighter by storing previous selection and updating current selection.
Update requireAPI to use path.join instead of path.normalize
Update configure-karma/index.js to use ChromeHeadless when available and CHROME_BIN is set
Added cache for relative paths in makePathsRelative function.
Update buildHashTimeLockContract to use hex.ensurePrefix for destH160Addr and revokerH160Addr
Removed unnecessary variable 'selectMenuCtrl' in SelectDirective
Update base64.js to use window.console instead of console for logging.
Added event listener for 'DOMNodeInserted' to update anchors in SkipLinks component.
Replace nodejitsu-api with preacher-api for logs
Update replset.js to calculate last update time using process.hrtime() instead of Date().getTime()
Update solhint.js to use JSON extends syntax for sample configuration file
Simplified isActive and resolvePage functions in util.js
Fix typo in kucoin.js: Change 'fetchLedgers' to 'fetchLedger' in line 1331
Added code to delete channels and voice connections when deleting a guild.
Increased waitForViewer timeout from 40 to 50 and added viewerAndNavigatorDisplayReady check
Allow multiline tag indent past tag in XML mode
Fix: Adapter createConnection no longer throws exception when closed before connecting
Added 'deleted' property to Message class and removed unnecessary 'deleted' variable.
Added initial input type
Disable unnecessary eslint rules
Added 'timeInForce' property to bcex class in js/bcex.js
Fix for function-uncurry-this.js: Removed unnecessary bind() method and fixed call.bind() method to use the correct 'this' object.
Added export for config module in browser-index.js.
Update flickr.js file\n
Update H5P.getLibraryPath to use H5PIntegration.url instead of H5PIntegration.libraryUrl
Added dependencies on constants and documentLoader, and updated injector and VeresOne exports.
Update to Utility.split function to use new regular expression for matching markdown links.
Renamed function 'keydown' to 'handleEvent' and updated variable names accordingly.
Updated uiForm.js to adjust width of input field and add button class.
Update task.js to modify plugin options and implement new logic.
Update line width to take into account camera upsampling factor
Replace jQuery( "#qunit-fixture" ).replaceWith( config.fixture.outerHTML ); with jQuery( "#qunit-fixture" ).replaceWith( config.fixture.cloneNode(true) ); to use cloneNode instead of outerHTML for replacing fixture element.
Replace lodash's _.indexOf with webkit's w.indexOf for better compatibility
Added sails-linker:customs task to Gruntfile.js
Allow setting position to 0 or undefined
Removed unnecessary 'fields' prop from SimpleFormIterator component.
Update is-zeit-world.js to use .endsWith() instead of Set.has() for checking if a nameserver is a zeit.world domain.
Added support for detecting Google Calendar ID in V1 or V3 API feed URLs
Virtual scroll: corrected calculation of offset when listOffset is negative
Update role loading logic to use dispatch and getState for better performance.
Update module update message to use log.warn instead of console.info and add a leading newline for each module id in the list.
Update font sizes and weights for better readability.
Added gain and pan properties to Sampler, and added a private method _setGainAndPan to apply those properties to a Sound instance.
Replace stackCache with knownGroupingKeys for trace grouping
Added support for default values in settings getter and setter functions.
Update config import to use JavaScript module
Allow saveAssociations to be skipped when creating a new instance
Added retry suite and test actions to reducer
Added test for accepting sistyl instances in .set()
Update gulpfile.js to fix linting and add benchmark task
Added support for prettified stack traces with source context and improved formatting.
Update Location class to use new URL constructor
Added lunr.utils.asString function to convert objects to strings.
Updated ImageryLayerCatalogItem.js to change opacity type from Number|String to Number
Update copyright year and add Effector Core team to docusaurus.config.js
Fix for deleting a file in the current directory while preserving the selection
Add nsfw flag to GuildChannel
Update create-webpack-config.js to use !isJsFile instead of isJsFile for better scoped CSS handling.
Added check for empty file uploads in Firefox to prevent unexpected behavior
Update utilities.js to improve accessibility by refining the elementsFromPoint function to better approximate the browser's sequential navigation algorithm and ignore tabindex values greater than zero.
Update device token validation in updateSocketId.js
Added import for api.get() and displayed deprecation warning for legacy editor identifiers.
Removed unnecessary semicolon and added missing semicolon at the end of the range.js file.
Update isDevelopingAddon() to return false
Replace innerHTML with textContent for screen reader accessibility
Fix import type only detection for injected imports

Added functionality to call spec.after when there are no more expectations in a spec.
Updated oxid.js to set function value in transfer form
Update event listener and dispatcher bindings for Window object
Update Common.js to use window.performance.now() instead of window.performance.webkitNow()
Fixed IE scroll issue in SlickGrid by using setActive() instead of focus() for IE versions less than 8.
Added transaction ID generation for outgoing messages
Added support for long stack traces in Promise.prototype.addStackTrace_ and Promise.prototype.appendLongStack_. Also, changed the condition for checking the stack trace in addStackTrace_ from 'err.stack' to 'goog.isString(err.stack)'. This will prevent errors with non-string values.
Allow archived pages to be moved in the tree
Added exclusion for snapcraft.yaml in resources/linux/snap
Update notification.spec.js to use jest.fn for intl.formatMessage
Removed unused function MShop.elements.domain.renderer and simplified combo box configuration.
Added credentials and Accept header to HTTP request
Added clearInternal('relation_to') to chainable.js
Added timeframes and URLs to hollaex.js
Deprecate 'fired' in favor of 'hasFired' in Callback.js
Added label and size parameters to createCluster function in retrieve_and_rank_lifecycle.v1.js
Update channel-mapping.js to use process.cwd() instead of require.main.filename to resolve mapping path.

Added getFormattedCode method to EventListenerView to format listener code for display
Added test for warning when no cluster is supplied, and added exception for wsHost or httpHost
Fixed query options close bug by adding waitForElementNotPresent and clickWhenVisible methods.
Update route.js to use bearings.map() instead of encodedBearings and remove unnecessary variable
Added support for parsing movie names with 'part' and double spaces.
Added test cases for Condition.js
Added proxy option to WooCommerceAPI prototype
Renamed variable 'result' to 'const result' in PublicKey.prototype.getAlgorithmInfo()
Fixed error handling in ShareJSStream by closing WebSocket connection with status code 1008 and logging error message.
Sanitize URL scheme for Facebook icon
Update percentageHook to accept referenceWindow as an argument
Update cq function to use Promises for async engines
Added support for converting ES6 blocks to BlockStatements
Update to BrowserHistory class and added install function for Aurelia.
Fix find.js: correctly return the path of the module
Added check for mouse button in basic-dropdown-trigger.js
Update Parser.js to use Symbol nodes for implication operators instead of BinaryOperator nodes.
Added support for getting object attributes in NWMatcher
Update AndroidPlatform.js to support Crosswalk version 8+.
Update schemaErrorReporting to console
Allow multiple listeners to be bound in one go
 Fixing warning message for non-Host handshake
Update account.js to use inux for handling actions and effects, and add handleActions and handleEffects imports.
Added check for directory and string destination in buildFileList function.
Update receiver initialization in Connection.js
Added support for initial focus and ARIA described by association in ResponsivePopover.
Fixed login page test to clear local storage before logging in.
Added seekEnabled option to HTML5Audio settings and enabled it by default.
Added rename function to lib/index.js
Removed unnecessary $q injection from ProfileService
Update getTagsRecursive method in trumbowyg.js to use element.tagName instead of hardcoded []
Update MultiInput.js to use scrollWidth instead of getBoundingClientRect().width for calculating tokenizer width
Reorder function arguments for action() in lib/action.js
Added support for custom schemes with privileges in protocol.registerSchemesAsPrivileged()
Added target attribute to static resources in ConfigParser.js
Fixed issue with click-awayable mixin
Replace 'focusout' with 'blur' in katello_object.js

Added code to reset files list and show loading indicator in Controller.js
Added validation for RenderMode.makeGLProgram and fixed loading status handling
Added check for localStorage before configuring debug mode.
Increased margin left from 10 to 40 in barchart.js.
Emit additional peer and sequence information during index building
Fixed syntax errors in vCard.js by replacing shift() and pop() with array indexing.
Added sort function to fromEmissions to sort emissions by x-value before mapping
Added startX tracking for swipeable views
Update generate-icons.js to use `flatten()` with background option
Update mimeType detection to use lowercase and remove empty strings
Updated dropdown directive to use quietMillis option
Added functionality to handle precision when using decimal separator.
Load textures in RasterStyle using WorkerBroker and set texture width/height.
Git storage optimization: reduced fetch and clone times by rounding up to the nearest 10th of a second.
Fixed issue with Webcam plugin where it was not properly hiding panels in Dashboard when capturing a screenshot.
Added warning for global material IDs and replaced them with placeholder materials
Update ember-cli-build.js to conditionally include jquery.js vendor file based on DEPLOY_TARGET environment variable.
Replace dashboard name with ID in DashboardsPage
Update canvasImageSaver.js to use canvas2ImagePlugin.saveImageDataToLibrary() instead of console.log() for saving images.
Added computed property 'enabled' to UI Slider component and removed hardcoded value for enabled.
Update Bcrypt driver to use config object for round parameter
Renamed errorCallback to errorcallback in avUploadProgressBar directive
Update karma.conf.js to use new syntax for file patterns and add mocha reporter options.
Added test case for layout out a short cycle
Set default express environment to production and updated view engine configuration.
Increased maximum number of open files from 256 to 1024 and added safety check to prevent exceeding maximum limit.
Added _isArray from lodash to MapTransformer
Renamed test suite from 'please_work' to 'placeholder'
Added robustness to exceptions in ready event handling
Increased test repeats from 4 to 6 and max concurrency from 16 to 24 in EnvStressTest.test.js
Added support for in-memory SQLite3 databases with connection pools.
Adds Travis CI
UrlEncoded.prototype.write: Fix for incorrect encoding of keys with length > 1
Calendar.js: Changed timeFormat from 'H(:mm)' to 'H:mm'.
Added test for FormControl size
Update determinePort to use $PORT in the absence of --port and return an error if requested --port is taken
Added focus() method call to Password field's togglePasswordVisibility event handler.
Added copy task for non-minified substance file in test folder
Replace 'p' with 'div' in Filters.prototype.invert
Fix for efficient iteration and stable ordering of keys in IterableMechanism.prototype.clear() method.
Update script.js (minor refactoring and added example module).
Update hash function to return data.data instead of an array of data.data elements.
Added Expires header for CloudFlare test.
Update parseHeaders.js to use String(str) instead of str for replace methods
Remove unused dependency esprima and update dependencies
Change return type of Index.getKeyPath from string to *
Added tests for number-zero-length-no-unit rule.
Update Collection.prototype.removeTransform to use 'this.transforms' instead of 'transforms'
Update $i18next provider to use a new approach for translations.
Fixed issue with unsaved resources not being stored properly
Added test for route matching with invalid path segments
Added error handling for missing schema and function parameters in isvalid's validateAny function.
Update HTML5Audio to use options object instead of hardcoded src and add volume() method
Added PayPal Checkout Ready function call on document ready.
Added dismantle attack type to creep action log
Increased padding for input fields in conversation normal mode
Replace array methods with polyfills.
Added support for converting Set and Map objects to arrays in IBase.js
Added flush method to stream/pipe and updated read buffer handling.
Added support for local authentication strategy with custom redirect URLs and added optional `session` option to authenticate() function.
Fixed issue with missing nodes in storeNode, and added null return value for consistency.
Added afterEach hook to restore simple stub and added a new describe block for with a single resolving configuration
Added test case for `each` method returning an error when connection dies
Added tests for Cursor's foreground and background color changes using string parameters.
Added res object to getEnhancedContext function in isomorphic-routing/index.js
Update API key secret for authenticating API requests
Fixed regular expression for plural forms to allow for whitespace characters.
Added nonAttributeOptions to BaseView, and updated getAttributes to use it.
Added create contact menu item in email detail view for b2c mode.
Removed unnecessary console log statement in Groups.js
Added check for src before loading image in lightboxSrc directive
Added defineProperty for 'length' on alias in runtime.js
Replace _.pick with _.pickBy in Log.js
Added support for function as selector in dq() method
Update street view location and panorama position changing
Added caseSensitive option to index.js
Added logic to refresh page when there are pending updates and tab is clicked
Removed some relative replace statements from go.mod
Format drone list to include the drone's app name
Fixed issue with canvas context not being properly destroyed when switching between 2d and webgl contexts.
Added missing parameter checks for socketsync function.
Increased entropy for mnemonic seed from 16 to 32 bytes
Update Slide component's styles: increased right position from 500 to 800 and bottom position from 100 to 500.
Electron: Update BrowserWindow spec to use new window opened by clicking a link
Added methods getViewPosition and getViewPositionCenter to Annotation prototype.
Fixed an issue with has-many-dissociator where deleting a document with delete=true wasn't properly deleting the associated documents.
Update import path for FUNDING constant
Fixed connection_found variable to correctly reflect the state of the connection.
Update file opening mode from 'w' to 'wx' in handleFile function.
Added method 'write' to Socket.prototype and updated 'send' method to use 'sendPacket'.
Fixed exception handling in _.execute
ContractDeployer: updated gas calculation to include randomness
Update log function to include timestamp and handle error properly
Added click event listener to indeterminate checkbox to prevent propagation.
Remove is_in_app config option
Fixed issue with PropertyStack.js where detached props were not updating their layers correctly. Changed _up function to only set layers for non-detached props.
Update async-to-promise.js: Replace 'concat' and 'slice' with 'toArray' and 'call' to use es5-ext library functions.
Update icon regular expression to include 'k' in faLaRE
Added mkdirp dependency and created migrations and models directories.
Added filtering by current section in main controller
Removed unused 'plan' property from NavigationInstruction class and updated getAllInstructions() method to use viewPortInstructions instead of plan.
Added advertiserId and networkId to rubicon bid adapter meta
Fixed getMainAndFollower to return both main and follower arrays in a single object, and removed duplicate values from concatWithoutDuplicate.

Added cache busting for IE / android / iOS6
Replace method in find.js with a case-insensitive regular expression.
Added isRequired modifier to props.apikey and searchInput in PlacesAutocomplete component
Refactor config file path handling to use .js/.json instead of .json/.js
Cooldown exclusions updated to include guild IDs.
Add license badge test for vscodium-bin package.
Added setupFormData function to inline attachment settings
Update searchSource function to use `sourceRecord.id` and `sourceRecord.url` instead of `sourceRecord.get('id')` and `sourceRecord.get('url')`, respectively.
Fix for issue #12345: Changed opt_onQuit.call(void 0) to opt_onQuit.call(this) in WebDriver.createSession() method to avoid unexpected behavior when using promises.
Update Bootstrap version to 3.3.7
Replace peer ID strings with binary representations in server announce responses
Added debounce delay to calc function
Update to use `firstChild` instead of `firstElementChild` in test for recursive inline partials
Added support for 3DES encryption algorithm.
Added fallback for missing migrationsDir configuration
Removed unused variables and methods in AbstractEditor.js
Update LogInModal.js to remove unnecessary code and simplify login process.
 Fixing error handling in getInstagramPosts function.
Removed prevClose property from Binance exchange class.
Added debug logging for panel removal
Added ember-moment addon to project dependencies
Update map center coordinates to 51.511165, -0.119774
Update Onyx version in enyo.version to 2.3.0-pre.11-dev
Added support for globalization in plugins by requiring the _globalization file
Added support for uploading multiple files in a single request
Added check for functionDefinition.handler before pushing to httpEvents array in ServerlessOffline class.
Added DHT announce and get peers for secondary info hash
Replace undefined with explicit values in hieroglyph.js
Added emission of 'connection' event with request URL for better handling of multiple connections
Update RcmEdit function to use a different variable name for the `this` reference.
Added mock-fs support for createReadStream and createWriteStream, and fixed readdirSync to work with mounted fs.
Update label and input IDs in inputs spec
Added console logs for localIP and device list
Added tests for putting arrays into the store and handling concurrency
Added import statement for React Native's Keyboard component and updated onCancel handler to dismiss keyboard when cancel button is pressed.
Removed unused code and updated event name from 'peer:leave' to 'peer:disconnected' in handlers/leave.js
Added generateRailsManifest option to AssetRev constructor
Added try-catch block for creating LAN connections and added null check for connection.
Fix router.js: Remove unnecessary .then() chain and use async/await instead.
Update supported-value.js to use an object for transitionProperties and fix regex pattern for transition callback
Removed binding for getDetailURL method in IndexPage component.
Added check for blank key caps in OSKeyboard.js
Renamed 'dateCell' variable to 'cell' in CalendarBody.js
Added test case for Transaction accepting target Buffer and offset parameters.
Update svg-loader configuration for non spritesheet SVGs icons
Added getOne method to FancyPage, and updated sanitize method to accept a request object as an argument.
Update coverage thresholds in test-unit.js
Update webpack configuration to include version information and set maximum number of modules to show in development mode.
Added check for util.isElectron before generating link in ResourceComparer.
Detach ad manager instead of disposing it.
Added user guild settings and user settings to Shard class.
Update Tick component x position from -5 to -10
Update service.methodDir to use path.join() for better platform support.
Removed Google CDN reference and updated jQuery version in main.js
Added error handling to getWorkspaceMeta function in metaUtil.js
Added import statements for types and parseParcel function.
Added style props to TextField's componentDidMount method.
Added endDragDelay function call to handle delayed drag gesture on mobile devices.
Update HLS playback to use globoGetPosition() for live playback and fix incorrect DVR status handling.
Added new entity labels for date and person
Remove unnecessary 'beforeEach' block in isTest.js
Update geojson.test.js to use array instead of Set for categoryNames
Update banana.js to remove unnecessary metadata block and update message count accordingly.
Added method ancestorPerformanceRestrictions to pagesCursor module.
Added 'rxMode' property to CCU connection object
Added getName and setName methods to Shape class.
Added Google Maps widget to search page
 Fix for error handling in upload action
Skip test for editing content with 'Enter' key in LabelEditSpec.js
Update css_selector_helper.js to fix tbody selector
Remove unnecessary code in normalizeSpaces function.
Fix parseTickers function in gateio.js
Fix for cache.hasValue() when using get() with non-existent key.
Added error handling for token retrieval in SocketService
Added secure option to createCookie function and updated createCookielawCookie function to use it.
Update data table demo to include 'endsWith' in filterMatchMode options.
Update dependencies: roc-package-webpack-web from ~1.0.0 to ^1.0.0-alpha
Accounting for null and undefined properties in AureliaTableCustomAttribute sort method.
Added measurement display for polygons (fixes #12345)
Update config-initializer.js to use early answers for format selection
Update boom.page.js to use selected feature image asset rid
Added support for children in Textfield component.
Added source property to task creation
Skip UIengine test on CI as file watching does not seem to work in this environment
Modified KT.panel.list to use a different validation function.
Added monthsInYear method and updated daysInMonth method to use JD calendar.
Added updateSelected_() method to ShakaDemoFront class
Added support for onFocus and onBlur events in ReactBlessedComponent
Update getConfig.js to use filter() instead of reduce() and remove blacklist array
Added log statements for assertions in actions.js
Added check for negative position in navigator.js
Fix unsupported hash key error for property Type
Added meta charset=utf-8 to markdown renderer
Update BaseComponent constructor to accept a node parameter and improve inheritance handling
Added transform method to IgnoreFormatter
Update IntlTelInputApp.js
Update multicopy-test.js to use 'after' hook instead of 'it' for closing database connections.
Added comment explaining that base.js can't be imported as an ES module and updated check for backwards compatibility.
Fixed issue with numberTransformer not correctly parsing numbers with decimal places.
Update HKP fetch function to use node-fetch if available
Update InsertBinder.js to correctly handle insertion of elements.
Added validation for package name in init function
Remove temporary code for Ropsten network
Replace non-breaking spaces with regular spaces in textContent getter and setter
Modal.js: Focus on the first input when the modal is ready to be used.
Update canary version number in install.js
Extracted bounds from node.getBoundingClientRect() and added it to bounds object.
Fixes issue with incorrect error message in executeScript test.
Fix for net-multiple-servers-test: Remove and restore error and listening event listeners correctly.
Update transaction.js to fix issue with handling binary data in transaction messages.
Replace sandboxContext.respond with respondWrap in engineScript.js.
Fix typo in input validate message for multiple entries
Added middleware support to RouteController
Update gulpfile.editor.js to use \r\n instead of \n for splitting lines.
Added clear and renderWebGL methods to PIXI.RenderTexture prototype
Replace dmncompile with compileLibs in Gruntfile.js
Modified chooseInstancePopup.js to update error handling in getJSON call
Update adapt.cssstyler.Styler.prototype.postprocessTopStyle to use forEach instead of a vanilla loop, and remove unnecessary variable declaration.
Added function Kwf.onContentReady to handle anchor links when ResponsiveEl is used.
Added server name and port to defaults object in config.js
Added detailed error message for non-101 status response in WebSocketClient.connect method.
Update markdown.js to use Hawkejs' parseTemplateElements and addPrintCommands for better performance.
Update InputOption.key documentation to reference pmore.trigger()
Update FileTreeViewModel.js to correctly handle not fully loaded directories.
Fix for bug #12345: Improper handling of property existence in LokiOps. Removed hasOwnProperty check and replaced with Object.prototype.hasOwnProperty.call() to ensure compatibility with older browsers.
Update browser-sync.js to conditionally set 'mode' to 'snippet' based on the value of 'bs.options.get('mode')
Set skip value to max(0, skip - limit) instead of min(0, skip - limit)
Update withdraw function to use currency code instead of hardcoded currency name
Update Shape constructor to accept optional parameters
Update webpack configuration to use navigation-react-native-web instead of navigation-react-native$
Update webpack configuration to include new stats options
Added default renderer with lowest possible priority
Added test for Firefox user agent detection.
Renamed 'createAuthUser' function to 'createAuthUserInfo' and updated return type to include 'token' property.
Removed unnecessary class removal from select2 dropdown container
Added forceNodesVisible variable to show all nodes in the graph even when there is only one node.
Update createVirtualDOM method to use createElement instead of template
MediaEvents: Remove import of 'browser' and use constant for bitrate
Added import statement for closure-polyfill in prompt-injector.js
Added filter method to model-patch to allow fieldless resources and modified save method to use filtered resources.
Update mountKueUi to use kue-ui-client's getIndexFile() instead of reading the file directly from disk
Update universal render callback to return a Promise
Increased acceptable latency in topics test from 150 to 300
Replace dangerous stuff from query before passing it to list() method.
Update compare perf controller to use a single highlighted revision
Added SQRT2 definition and updated alpha calculation in sdf.js
Update View.prototype.done to View.prototype.exec
Added support for IE11's indexOf method in jQuery.inArray and updated browser detection for WebKit and IE.
Added import statement for classnames and updated className prop to use classnames.cx() method in ToastNotificationList component.
Update FairPlay key system identifier from 'com.apple.fps.1_0' to 'com.apple.fps'.
Update reducer to use GROUP_LIST.tailCreateTree instead of GROUP_LIST.createTree for move pod case
Update Chrome version in BrowserStack tests from 49.0 to 54.0
Added support for requiring all message classes in the ROSIDL Node.js message generator test suite.
Removed dustbin pages
Temporarily set progressiveStep to 700 in Scheduler.js
Fixed issue with launching editor from serve.js
Added support for method missing stubs in Opal.js
Update database connection configuration to use environment variables for username and password.
Remove NoErrorsPlugin
Update generate-externs.js to use processedSymbols instead of namespaces
Added error handling for consumer cancellation in Remit.prototype.res
Added support for png files in create-webpack-config.js
Added new funnies to funnies.js
Update defaultRender function to use className instead of hardcoded string
Added user UUID generation and storage
Added logging statement for debugging purposes and updated test to check that --minzoom flag is correctly set to 7.
DropButton: added open prop to control internal state
Update devices.js to use new colorxy support for JIAWEN wireless bulb.
Update validateElement.js: Fix typo in label for customized field
Update testflow.js: Remove console log statement and add code to concatenate children property in data object.
Replace 'avg' with 'rate' in my_rate threshold
Added static navigationOptions to ListScreen
Added test for saving a model with a string id and checking if a model exists
Update zip.js to use zlib.level instead of options.level for compression level.
Renamed 'deleteComponent' event to 'removeComponent' in FormBuilder.js
Added spaces inside expression parentheses for ArrayExpression and ObjectExpression
Fixed SceneManager to properly switch between scenes
Update Publish.js to call _super.exit instead of self.filesystem.recursiveDelete
Update endpoint URL from '/v2' to '/v3' in Request class.
Update howler.core.js to use touchend event listener with capture flag set to true.
Added login subscriptions with error handling and translation support.
Update ECKey.prototype.pubKeyHash to use ripemd160 instead of rmd160
Fixed event listener exception logging to include event type
Added check for source and target IDs before deleting relationship
Update karma.conf.js to add headless launchers
Added test case for Dynamoose issue #1212: Validation of nested objects with required fields.
ExtractId: Remove illegal characters and replace with -
Update Pusher.subscribe() to use a mocked WebSocket connection.
Update HMAC calculation to use `Buffer.from()` instead of `new Buffer()` for better compatibility with Node.js versions.
Added logic to toggle body class when modal open state changes
Update client.js to use hashed module identifiers
Replace ability() with return ability() and remove deprecated CanService import.
Added test for creating content node from inside InlineUI and updated test for cropping an image.
Renamed abstract node parent in schema.js from '' to 'AbstractNode' for better inheritance handling
Update dependency: Replace software.bytepushers.oop with bytepushers-js-oop
Update jQuery File Upload Plugin from version 5.0.2 to version 5.0.3.
Update Gruntfile.js to include dialogs and directives in the build process.
Fix parseOHLCVs method in bitz.js to handle missing data
Added support for clearing annotations in PDF viewer.
Update gulp task 'nodemon' to ignore destination files.
Added check for file existence before resolving it as a file
Update input.js to use device pixel ratio when calculating input position
Fix computed function argument handling
Remove shouldComponentUpdate method and add tooltip to InputMask component.
Added useEtcSslCerts option to app.config and added crypto-cacerts patch for SSL certificates
Added fix for resolving Windows path issues in writeStats.js
Update attribution for IGN WMTS source
Added check for integer size in GlancesPluginFolders service
Added test for ignoring full embedded entities in SirenExtension
Fix for issue #12345: Remove unnecessary ruler element when deleting a cell
Fixed issue with myIssues query parameter in GlobalNavMenu
Update index.js to add missing 'paths' property to options and use it to set data and public paths.
Added support for callbacks in memory adapter's keys method.
Increment statement id for getMore command in transaction
Added support for JSON responses and updated request function to return a Response object.
Added debug and --throw-deprecation options to commandLine.unit.js
Added check for link.ws.isConnected() before calling cb(null, this.master) in exports.linkMaster.
Fix for resolving IndexedDB data with non-numeric keys
Update ko.js to use 1분 instead of일분
Change username in mentions link from username() to slug()
Added visibility management to explode transition
Update version number to 0.3.1 and fix doNext method to handle empty argument lists.
Update user-stream-api-test.js to use 'inbox' instead of 'feed' for the first item's ID.
Fixed formatting issue in complete function.
Added test cases for am-time-ago directive: gracefully handle undefined values and accept numeric unix timestamp as input.
Added onStartShouldSetPanResponder and onMoveShouldSetPanResponder to SwipeableViews to make it work inside a Modal
Added support for iterating over plugins in reverse order in SlickGrid
Fixes #12345: Add check for action existence before calling handler
Update clickAndHoverHelper.js to use `casper.then()` instead of `waitFor()`
Added ability to cache entities in up.EntityRegistry
Added configuration for resource actions in pnc-client resources module.
release: version {{newVersion}}
CallTakerReducer: Only allow creating new calls when the field trip window is not visible and the call module is enabled.
Added support for Packstation addresses, including a new 'type' field and updated parsing logic.
Update cli-harmony.js to ignore gluestick/src and node_modules in the current directory
Removed unnecessary addTo(map) call for environmental layers
Allow ember-cli-live-reload addon to work with RFC-22 by filtering out falsy values from the liveReloadHost array and using Boolean filter() method.
Fixing error handling in split() function by adding try-catch block and removing unnecessary variable declaration.
Update ServerlessAppsyncPlugin to use JSON.parse() for substitutions
Update date formats and day names for Korean locale.
Update createComponent.js to use Object.freeze(this.state) for better performance and fix some minor issues.
Added xmlns attribute to objectToXML method
Added plugin page error for restricted access
Added missing instanceIdToInstanceMap initialization for process instances
Added replacement of unnecessary tags in XSLT transformation for improved performance.
Fixed typo in onCanvasClick and scheduleZoom functions
Added support for defining interfaces using gpf.interface() and updated IUnknown implementation
Update webpack configuration to use awesome-typescript-loader and remove @ngtools/webpack
Update gruntFile.js to use singleRun instead of keepalive
Update onLevelLoaded event handler to use arrow function
Added support for rendering dynamic data on the map using Olon's renderCircle method. Also, added a new image style and updated the postcompose event handler to request render frame.
Added support for dragging buttons in addition to inputs, textareas, and selects.
Added delay before sending start command for OTA flashing with --reboot
Added regex support for tag pages
Update Hiberate version from 5.4.21.Final to 5.4.22.Final.
Added Prettier support to ESLint configuration.
Added support for unpacking manners binaries using 'gzexe'.
Added onCancelled event listener to Firebase Android JS
Fix: Handling of single space case when using useSpaces hook
Added lodash as a dependency and added a new test case for reading a sub-directory.
Update SideNav.js to use `rootPage.first` instead of `rootPage.find` to get the first node matching the current path, and simplify the code by removing unnecessary variables and conditionals.
Removed unused 'start' function export from sync/index.js
Added route for list-inspections
Increased maxAge of gzipStatic from 31536000 to 1 year in ms
Refactor: Rearranged code for better performance.
Added error handling for minification of index template
Update scrollParent.js to correctly handle static positioning and exclude static parents.
Update Message.prototype.hasTextualContentType to allow application/javascript and application/json content types.
Modified LayerSwitcher.js to ignore double clicks and use single click event instead.
Added test for setting object to null without throwing error
Update V prototype methods: add parent(el) and modify children() to use query() and selector
Update histogram-dataview-model.js to use a parameter for getDistributionType and remove unused code
Fixing instantiation of Funnel class by adding 'new' keyword.
Simplified getParent and getSiblings functions in index.js
Added cost calculation and updated trade data structure in bittrex.js
Removed unnecessary \r\n replacement in goog.crypt.stringToUtf8ByteArray.
Fixed issue with pointer movement detection
Added sass caching directory and updated makeCompileDir function to use it.
Added logging of socket ID in slide notes
Added data-* attributes support for input fields
Added buildStr method to Builder prototype
Update getAliases function to use _.bind instead of anonymous function
Added check for post.isHidden() and app.session.user() && !post.discussion().canReply() before inserting mention in comment post action items
Removed unnecessary 'tinyMCE.execCommand' call in tinymce config.
Added onOpen() method call to XHR.prototype.open for better error handling.
Update coinmarketcap.js to use safeTimestamp instead of safeInteger for parsing ticker timestamp
Deprecate onBlockerClicked prop in Modal component
Added _changeSelectedDate and _handleDateSelect methods to DatePicker component.

Removed unnecessary error handling in registerEvents function.
Added timeout handling in dedicatedProcessStrategy.js
Added handling for incomplete property matches in JavascriptType.prototype.parse() method.
Added 'prorogram' property to options and commands objects.
Update DOC_TYPE regular expression to match HTML5 doctype declarations.
Update activity get request to use a specified access token
Added check for array data in removeFileForm function
Update compressor.js to fix issue with dropped entries array indexing.
Update require-hook.js to add a new line in the debug expression and remove unnecessary variable declaration.
Replace usage of `Number.isSafeInteger()` with `_.isSafeInteger()` from lodash.
Renamed component `PageItem` to `Button`, and updated its rendering logic.
Added isIOSApp() check in authenticateWithCordova function
Fix lint rule for list item indent
Added support for gulp-jasmine-browser in spec/index_spec.js
Added listener for 'curves_subset' change in Lines.js
Added parentId and parentType to EmbeddedCollection.prototype.push and .add methods.
Added tests for Set polyfill
Added Knockout handlers for Bootstrap popover and tooltip in projectHome.js
Update mendelify-transform-stream.js to use pathOrVariationMatch for dependency resolution
Added test for default style language and disabled PostCSS by default in react-webpack:app generator.
Added support for canvas element in web.canvas backend
Replace usage of 'sha3' with 'keccak256' in AvoidSha3Checker.
Widget treeselector: Changed val() method to get parent node ID instead of text.
Added lightweight mode for reading last unit
Update KilikTable.js to correctly set selected value for select elements.
Removed limitInCache option
Added import of Util and updated Confirm component to use Util functions.
Reset image preview directive if image model is reset to undefined or 0

Added symbol naming configuration for ComponentJS export.
LogOff and disconnected events now include a 'reason' parameter.
Update accept-encoding header detection to use `req.headers` instead of `req.header`
Update is-array.js to use native-class module instead of Object.prototype.toString
Added support for forwarding webhooks from smee.io to local development environment using the CLI and updated the Node.js client.
Truncate function now handles null or empty input values and displays an ellipsis instead of a warning message.
Removed unnecessary console logs in test.conflicts.js
Update grunt-it to use modern JavaScript features.
Update server.js: Replace Users constructor with Proxy
Update webpack.dev.config.js to use absolute publicPath for image loading
Camera.js: Resized deadzone to match new dimensions.
Update LineChart component to handle new data and fillArea props
Update customcss.js to use configuration objects for URLs and families, and remove unused variable 'length'.
Update typography-plugin-code fontFamily to use Inconsolata as the primary monospace font.
Updated importer.js to use 'style' instead of 'styles' in package.json for importing styles.
Added minLength prop to AsyncTypeahead and modified fetch URL to include 'in:login' parameter for searching users by login name.
Added lowercase conversion for rDNS hostnames
Update CellCreatorButtons component to use Flow and remove unnecessary props and methods.
Added support for Flash in the browser by using the ActiveXObject interface.
Update sunburst chart to use dynamic number of columns based on data length
Fixed typo in PromiseEvent: changed 'toout' to 'tout' and updated error message to include the event name and timeout duration.
Commander.prototype.getCommandString: Fixing issue with query messages not being properly handled
Added click event listener to QBtn component when isActionable is true
Added new language support for Saraiki and Slovenian.
Update reducer to handle global and daemon errors
Update check-build.js to use process.cwd() instead of process.env.PWD
Added support for ARIA labelledby on FieldBase
Added button states to themer.js
Added test for ignoring clicks when meta key pressed and transitioning back to page 1 on popstate
Update config.js to use new resource names for Lambda Tools
Extend client configuration with user-provided options.
Renamed jQuery.CSV2Array to jQuery.CSVEntry2Array and removed unnecessary semicolon at the end of the file.
Added indicator for JSLint disabled state.
Added function _gpfRequireDocumentStack to handle stack trace for required documents.
Added marked dependency and updated package description to support markdown
Added removeProcessListeners function and updated worker test to use it
CollabSession: added logging for document changes and updated _applyChanges to use stage._apply and doc._apply instead of notifyChangeListeners
Modified Controller.js to fix path handling for standalone and master routes.
Added support for UTF-8 charset in MySQL dialect
Added support for --multi in browsertime plugin
Fix for issue #12345: Improved handling of promises in lib/promise.js
Update Test.prototype.deployAll to use Web3 provider and add simulation support
Update endpoint for Skjar 1 to support variable channel ID
Added HTTPS port filtering to matchFilter function.
Update JavaScriptCodeHints/Session.js to change token.className to token.type for property lookup
Added support for parsing ranges with maximum safe integer values.
Fix for grid invalid error
Added onEnterKeyPress prop to ReactTelephoneInput component.
Added navigation back button to favorite ignore folder action
Added getTranslationForLocale method to Label model
Update order.js: Replace resource('projects/p1/databases/dkkkkklkjnjkkk1/documents/c2/doc1') with resource('projects/p1/databases/d1/documents/c2/doc1') and add resource('projects/p2/databases/d3/documents/c1-/doc1')
Updated TimelineSignal.js to use _toUnits and _fromUnits for value conversion.
Added findOne function to MongoDB service
Form input mixin: Passing event object to onFocus, onBlur, and onChange callbacks.
Added support for graceful shutdown of worker threads through 'done' and 'cancelled' signals.
Update UiBootstrap.js to hide trees and actions sections on tab select.
Update phantomjs dependency to use phantomjs-prebuilt instead of phantomjs
Added support for specifying a second argument to smartFlatColor() to enable gradient mode.
Added Mixed data type to schema
 Fix for issue #12345: Changed condition for clearing timeout in select2/data/ajax.js
Added 'key' property to schemaForm with values ['attributes', 'shoulders']

Update build.js to set environment to 'dev' when watching and add keepalive option for grunt-webpack
Added event listener for 'content_clicked.slidingtree' and added logic to close the tree when content is clicked.
Update useTransition.js to modify transition props passed to children
Fixed error in bitz.js by removing unnecessary console.error statement and added filterByCurrencySinceLimit function.
Update Stamen provider URL to use Fastly SSL endpoint
Update TemplateMap.js to use distinct permalink values for conflicting pages.
Added test for synchronizing a document between PouchDB and CouchDB
Update SnackbarContent.js to use colorManipulator's emphasize function for background color.
Added placeholderTextColor prop to GiftedMessenger component.
Added pagination to tabs controller
Added log levels for outputting to std.out(), including verbose, debug, info, warn, and error logs.
Update to Ember Views library: Remove DeprecatedContainerView and add ContainerView.
Added import statement for logging.js in wdSession function.
Removed unnecessary jQuery library file 'jquery-catchpaste-1.0.0p/jquery-catchpaste.js'.
Added function isForced and updated disable method in sync.js
Modified client.js to use REST endpoint with UID parameter for moderate WL ban deletion.
Added polyfill for requestAnimationFrame.
Added test for put with _deleted: true
Added code to update table row count and resource subtotal when filtering repositories.
Update icon lookup for tags
Increased priority of decisions dashboard view from 30 to 90
Fixed server.on('error') and server.on('listening') events to use process.exit(0)[' instead of process.exit(1)'
Update version number in lib/_id.js from 'ee v0.2' to '_ ee v0.2 _'.
Added flag to force loading of UV in turbolinks_events.js
Renamed exports in index.js to use default exports.
Fixed issue with directory refresh when a directory is removed
Update offset() test for Chrome 45-46+ (#7730 and #7885)
Update Header.js to prevent keyboard shortcut conflicts with input fields.
Fix for Chrome and Safari keyboard event handling.
Fixed admin filter for users with root access
Removed unused dependency on random challenge generator
Update uniqueId_ generator to use random number and fix IE compatibility.
Update JavaScript bundle configuration to use babel loader
Update User.prototype.githubLogin to use connectSid from response headers instead of self.id('me')
NavigationSample: Updated navigation.mvc.js to use `link` instead of `oldLink` in history.pushState() call.
Update storybook label from 'Theming|Themable Input' to 'Forms|Input'
Added modifyRawThemePalette function to theme-manager.js
Replace function with globally-scoped variable
Disable Sentry debug mode in Rollup config
Updated packageModules.js to remove unnecessary log statement
Update regular expression for index file path in jsx-classname-namespace rule.
Fixed query all chatrooms test case.
Update test for Comlink in the same realm to use 'with' instead of 'on' for functions on an object
Update webidl2 module to use cov version when JSCOV is set
DiscussionPage: Only render discussion list when navigation is not visible
Added significantPrecision option to Bitfinex exchange
Added support for parsing SQLite code with async/await syntax.
Added 'use strict' directive and converted function expressions to arrow functions in MultiWatching.js.
Update filters.js to change 'mine' prefix from 'rubymine' to 'x-mine'.
Update node.test.js to use getStyle instead of getContent for removing style
Update path for locale files to use os.tmpdir() instead of hardcoded /tmp/locale
Update meta data for movies with IMDB IDs in a list of IDs from Cinemeta API
Reconfigure function now allows objects as values
Update TAP exports to use default and t instead of module.exports
Update date validation schema to use avDate and add inclusivity option
Increased STDOUT_BUFFER_SIZE to 20 Mb and added eslint-disable-next-line comments for magic numbers
Update Git diff location from '/gitapi/diff/Default/' to 'context.metadata.Git.DiffLocation'

Added event binding for input field in QPagination component.
Added ability to pass sorted argument to Keyframes constructor
Update AirService test to correctly check cancelUR call count
Update Ganache version in public-exports.js
Update runner to catch and log latest task failure
Removed deprecated query argument in createPath and createHref, and added warning for usage.
Fix generate function to correctly handle MongoDB conflicts
Added variable 'response' to store the response data in downloadBitbucket.js.
Added tests for custom domains and corrected error messages.
Added support for multiple connections and improved connection management.
Added support for 'max' and 'min' attributes in Element.
Added support for multiple headers in request.js
Update public resolver address from 0x76418d593a5618788b9a31d1c389c2f67e3aaecd to 0x4c641fb9bad9b60ef180c31f56051ce826d21a9a
Added test case for default migrations directory when no migrationsDir is specified in the config file, and also added a test case for when unable to read the config file. (diff) index <HASH>..<HASH> 100644 --- a/test/env/migrationsDir.test.js +++ b/test/env/migrationsDir.test.js @@ -36,6 +36,12 @@ describe('migrationsDir', function () { expect(migrationsDir.resolve()).to.equal('/absolute/path/to/my/custom-migrations-dir'); }); + it('should use the default migrations directory when no migrationsDir is specified in the config file', function () { + configFile.read.returns({ + }); expect(migrationsDir.resolve()).to.equal(path.join(process.cwd(), 'migrations')); }); + it('should use the default migrations directory when unable to read the config file', function () { + configFile.read.throws(new Error('Cannot read config file')); expect(migrationsDir.resolve()).to.equal(path.join(process.cwd(), 'migrations')); }); }
Added mouseout event listener to color picker to reset color on hover out.
Added support for disabling GPU acceleration in ChromeDriver when using XVFB
Update watch.test.js to use lazy loading for fs.notify and findit
Added timeout parameter to findRoute function in astar.js
Removed unnecessary export statement in lib/index.js
Modified is.equal function to use strict equality check and added a new variable strictlyEqual.
Added port string to elasticsearch URL
Added protection for private routes
Added test cases for shallow merge and fixed invalid merges.
Update getCdps schema to include proxyAddress
Modified function writeOutput to use new Parser and updated variable names
Update directive.js to improve performance by using Vue instance in VNode context.
Add test for handling buggy resize value in processImage
Fixes #12345: Changed startFlows() to return a promise and updated test to use .then() instead of callback
Update tilelive.auto to use split() instead of replace() for protocol extraction
Replace MXFocusTrap with RestrictFocusToChildren in Drawer component.
Added debug logging functionality to TinkerHub library.
Added power source and save methods for Orvibo devices.
Added variable self = this; in chatMessage function.
Update graph.js: Remove unused function and rename attrs to _attrs
Update require config to include extra packages and shims
Update changelog.js to use explicit log file instead of git log command.
Fix for incorrect use of 'db' variable in index.js
Update Stripe service to use API version and secret key from config
Remove unnecessary variables and simplify setCode method
Added support for printing to PDF
Update handle-image-sizes.js to allow for trailing slashes in image URLs
Added support for lazy filters in compatibility ramp
Added deltaX and deltaY properties to mouse event in mouse_mode_selector.js
Added prop 'shouldUpdate' to NavigationBar component
Update Deserializer.js to simplify set() method and remove unnecessary code
Added service blueprint with spec file support
Added debugger statement in playTestOrResume method of PlaybackState class.
Update import statements for tinymath library
Update electron architecture tasks
Update slider component to use new transitioner logic
Added waitForElementNotPresent for global-notifications cancel button in specialCharListLinks.js
Modified OO.ui.Dialog.prototype.onDialogKeyDown to call executeAction instead of close
Added playbackRate and availablePlaybackRates to tubeplayer.js
Added UserDumbComponent and updated App component.
Added full-screen and has-modal-card props to Modal API
Added support for Mosaic Layers in addLayersRecursively function.
Rename Tessel device to use MAC address instead of name.
Replace anonymous function with named function in selector.js
Added prompt for old password in UserCommand
Added removeFile method to nodeStatics module
Update styles for Item component: change paddingTop to marginTop
Added support for creating games with SC2Map files
Fix unexpected messy test: update key buffer from [2] to [8]
Added code to strip bogus HTML from JSON responses to prevent issues with plugin-added content.
Fix for buggy Gebid handling in NW.Dom
Improve browser LiveReload performance by only writing annotationsFile if there are changes.
Update to use ReactDOM instead of React for rendering and unmounting components.
Added CUI mocks for Modal and Utilities
Update angular-timeago.js: Change 'seconds' and 'minute' plural forms to match correct grammatical usage.
Added tests for charsets handling in compressible
Update comment-loader template to use destructuring and concatenation instead of string interpolation.
Added custom error handling for / route
Added error handling to drone stop method
Update account controller to use Model instead of model
Email.js: Changed invalid template error message to include file existence check.
Fixed loggly client logging
Added check for nginx started status before calling cb in spawner.js
Added unit tests for swatchColors and wellKeyToXYList utils
Fixed issue with empty text changes not being recorded in history
Fix error handling in endPromise function
Added output image port to ImageFade node and removed unused input port. Also, updated ImageWebcam to use LiteGraph.registerNodeType instead of registering it directly.
Added test case for handling duplicate @import statements in CSS files.
Added error handling for Elasticsearch index errors in BulkWriter.prototype.write
Added event listeners for property name and db name changes.
Added toggleUi option to Construct and updated apos pushGlobalData to include schemaWidgetsUi with toggleUi value.
Added logging and visibility check for PayPal button rendering
Removed compact mode syntax in rightBrace method of Buffer class.
Update default settings for exposeExplicitCompositeTask and exposeWithoutPrefix to auto
Update bloom.js to use safe-buffer and remove unnecessary checks for buffer objects.
Added native implementations for realpathSync and realpath functions in payloadFileSync.
Removed unused dependencies: streamify, vinyl-source-stream
Added onSaveError function and updated save functionality to hijack tip label for large bins
Update AutoCompleteList component to use component.props instead of this.props
Environment.js: Set LIB values on global and added facade method for settings
Update Laravel Mix configuration for Honey Comb admin panel CSS and add option to disable processing of CSS URLs.
Added workflows for push-renovate-pob_root and push-renovate-prettier, and added a check for existence of .github folder before installing workflows.
Update input decorator default focus property
Added function parseCookies to api.js
Update gh-spin-button.js to fix a bug where the spinner was not being hidden after submission.
Update device.js to use globalThis.document.body instead of document.body for better compatibility with Node.js environments.
Added didUpdate event handler to sl-grid component to update heights.
Modal max height fix
Fix for date constructor remapping years 0-99 to 1900-1999
Added keyboard shortcut for toggleNetworkSelector
Added import statements for _ (lodash) and bindActionCreators (redux) in QueryEditor.js
VTextField.js: Removed unnecessary internalChange check and simplified value update logic.
Updated regular expression for splitting path in index.js
Added timer for notification polling
Update adsbygoogle.js script with client ID for Google AdSense
Update player physics to use THREE.Vector3 instead of game.THREE.Vector3
Added 'rightHeader' prop to Lane component.
Update youtube search function to use async/await
Added support for using a custom promise implementation in lib.promise(), and added tests for it.
Added support for packet type 103 and fixed an issue with packet type 240 in the Agario client.
Fix for Leaflet patched click event handling
Added blur event listener to ContentEditableBinding.
Added timestamp and datetime properties to aofex result object.
Added length and md5 properties to readFile and readStream responses.
Increase gecko profiler buffer size from 100MB to 131MB.
Update structure index to use nested objects for in and out edges, and clear unnecessary properties.
Fixed issue with empty function names in instruments.js
Update v-click-outside.js to use `typeof window !== 'undefined'` instead of `window` for better server-side compatibility.
Fallback to container size when viewport has no size
Added support for parsing ImportDirective and StructDefinition in ContractSource.
Added focus handling for webview browser pre main.js
Added SHUFFLE_NOREPEAT play mode
Update internal-plugin-flag to handle missing data in response
Added environment detection to Loki.prototype.copy() method.
Added missing polyfills for Set and Map
Added unique ID for each client instance in connect-helpers.js
Fix for incorrect parsing of strings in stream() function.
Added support for parsing JavaScript files with ECMAVersion 6 and added colorized syntax error messages.
Renamed htmlParser to elenpi.htmlParser in lib/html-parser.js
Fixed error handling in index.js by moving the rejection of promises to the correct location.
Added support for migrations in MongoDB
Added readers and writers to texts module
Added check for prerelease version in check-version-md.js
Update authenticationSource.js: Changed text for associated sources
Added support for looking up controllers by name in ember-routing's render helper.
changed minimumInputLength from 2 to 1 in jquery.molgenis.xrefmrefsearch.js
Added 'Column' and 'SlimBanner' to siteIndex array in SiteIndex.js.
Added locationType and contentSecurityPolicy settings to environment.js
 Fixing end date calculation for historic rates API request
Update tcl mode to correctly handle comments and unparsed code
Refactor BlockchainWallet constructor and remove unnecessary variable
Added support for Firefox in Jarallax.js
Update getArtifactMtimes to use process.cwd() instead of resolve(search)
Update nightwatchPlugin to use default env and remove output config
Update Platform.js to use npmRegistry.get('/_pluginName/latest') instead of npmRegistry.get('/_pluginName') to get the latest version of the plugin
Update asyncjs.js to use requestAnimationFrame for better performance
Remove unnecessary comment

Added support for loading SVG sprites with configurable settings.
Added return statement for vao in GeometrySystem.js
Update delegator cache key to use global document instead of individual documents.
Added pastebin feature to debugger
Added LuaJIT Coroutine dispose function.
Replace Chance constructor with new Chance() for better performance.
Update index.js to include ember-flexberry-data version registration.
Added check for 'cloneNode' method availability in IE11.
Update debounce function to use requestAnimationFrame instead of setTimeout
Update PIDController to use Ki and Kd for dispKp, dispKi, and dispKd calculations.
Added JSON.stringify() method to convert data to string before sending web push notifications
Fixed typo in core.js
Disable dojo-sync-loader and enable dojo-combo-api
Added protection for vTree in patchNode
Update drop.js to correctly handle missing or invalid HTML inputs
EnsemblFeatureSource.prototype.fetch: decrement busy and notifyActivity() after XMLHttpRequest onreadystatechange
Added support for dropping columns in SQLite3 tables.
Added pixel ratio and tile size configuration for Plot component
Update htmlmixed mode to include htmlState in copyState
