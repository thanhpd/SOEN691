Update trackCore.js: Remove unused property and add default color for new tracks.
Fixed bug in ConnectionManager._cleanup
Updated ResponsivePopover.js to set button types to 'Transparent' for begin and end buttons.
Changing loop mode from PingPong to LoopRepeat in hand-controls.js
Added constants stub for fastfile
 Fix error in testReportWithEmptyEndDateBeforeStartDate test
Added istanbul ignore flag for untestable code in _gpfExit function.
Added logging for error cases in destroyRole() in spec/helpers/generic-role.js
Updated third-party.js to remove query strings from module URLs and use a new mapping function.
Added templateError and test1 tasks to Gruntfile.js
Refactored findModelNodes to use reduce() and improve performance.
IE < 10 support for dirty forms
Replace cordova/commandProxy with cordova/windows8/commandProxy in AccelerometerProxy.js
Update getFiles function to handle Webpack 4 entrypoints
Added Errors global variable and updated throw statement in deriveAddresses method.
Update require path for ui-layout
Added variable _VALIDATIONS_REF to store validation messages and updated validationError function to use it.
Removed unused variable gen_keys
Added file UUID to uploads and fixed path format.
Update jQuery.peity.js to use 'this.each' instead of '$(this).each' for better performance.
Change iframe selector from $('iframe') to $('iframe#editJob') in public/js/templates.js
Update callSite handling in asyncFunctionInitialized
Change animation type from slide to fade in RkPicker.
Update ancestor declaration for gradient filter
Added keyDown event handler to ValidatedSelectComponent to trigger save action on enter key press.
Fixed issue with incorrect selection of datatable rows when sorting and filtering.
Added 'data-id' attribute to region elements in WaveSurfer.Region
Added program.option() and updated command registration
Allow IconButton to accept SvgIcon children
 Fix for Quill's shiftRange function to correctly handle user-generated content.
Update selectors optimizer configuration
Added overwrites parameter to lines method in TextMetrics class
Modified LiveTab.pointerMove to use const and updated logic
Update AppPermissions.prototype.checkPackageContentsExists to use 'self' instead of 'this' for unit testability.
Ensure data and options are objects in Pushy.prototype.sendPushNotification
Modified data.js to change the implementation of the agent and archives functions.
Increased network update frequency and added smoothing for mouse movement
Added flag to executeScript to disable path optimization
Updated PlaybackState.js to export current running test instead of passing the actual test object.
Update script execution commands: Use 'node' instead of './node_modules/.bin' for webpack and add jest-cli coverage option.

Update function signature to use 'addUtilities' instead of 'addComponents'.
Remove unnecessary physics engine initialization
Added db and password to queueConfig object in Bull constructor.
Update options object with new properties
Update bundleWebpack configuration to use {configPath: options.webpackConfigPath} and add crossorigin: false for HtmlScript dependencies.
Update IRC client to use twitch.tv server and port
Update RangeSlider handle event handling to fire liveChange instead of change event
Added setCustomMarker function call in compile method of custom-marker.js
Icon: Update iconFamily condition for Ionicons
Update jest.config.js
Update server.js to use `done` callback instead of `callback` for pool destruction.
Adjusted slideshow height calculation to account for parent element's height
Added SmartThings smart siren device support
Updates to PublishFlowOptions component: changed button text logic to use publishOptions instead of publishType.
Added ancestor and children sorting to Container component
Update symbols for Windows platforms in cli-utils.js
 Update to observable.js: removed unnecessary options argument in toString() method call
Implemented frame reading and updated error handling in lib/frames.js
Added environment variables to config.js
Added sudo to default interface command for better permissions.
 Fixes #12345: Allow Redback instance creation with a redis client object instead of requiring command_queue and offline_queue
Updated editor plugin to disable/enable link button based on node name and added respondToNodeChange() function to LinkForm.
Added support for transceiverRequest in Peer class.
Renamed 'validationObject' property to 'validationModel' and added deprecation warning for 'validationObject' in FlexberryObjectListviewHierarchicalControllerMixin.
Update test specs and clean up preferencesKey
Added check for maxTouchPoints in navigator to determine support for touch events
Added support for Chrome event handling
Update test_modrewrite.js to use grunt-contrib-connect-modrewrite instead of Authentication
Update uncss.js to handle multiple files
Update chance.js to use `process.stdout.write()` instead of `process.stdout.write(`) for printing generator output.
 Fixes #12345: Improves tokenization by handling paragraphs correctly when they end with a newline
Update streamCompare function to accept options or comparison function as last argument, and remove unnecessary 'Promise' default option


Added support for highlighting text with a specific style.
Added exception for Ionic app clicks in gesture.js
fix: filter out falsy values in getVnodes
Added selector for getting connection queue.
WizardProgressNavigator: Update step z-index when updating current step
Updated admin/common.js to enhance dynamic tables by adding and removing rows.
Update typography theme Irving's base font size and line height
Update Generator.resolved from __dirname to __filename
Update SIFTSCIENCE_URL and add const for hasLoaded
Added 'before' option to VerifyStream, allowing for custom data processing before gunzipping and TarBuffer creation.
Update karma-minified.conf.js to exclude smart.elements.js instead of smart-elements.js
Update to use new ES6 import syntax
Remove unnecessary checks for textFillState_ and textStrokeState_ in TextReplay.drawText() method.
Update CacheEntry.prototype.get to include cacheWhitelist in the result object.
Update error message handling in NylasConnection
Added getLanguageId() method to NullLanguageMode class.
Added warning for missing packages in npm.js
Update dependencies for CorrelationService and DeleteProcessModelService
Update passport-vkontakte strategy.js to use _scope instead of options.scope for VKontakte OAuth scope
Fixing error handling in sendMessageProto
Update Text styles with new color and letterSpacing variants
Fixed version number in uWebSockets.js dependency
Updated statsporn.js with new regular expressions for command and help matching, and added cooldownTimers array.
Added support for Error.captureStackTrace() in StatusCodeError constructor.
Add test for StyledComponent creation order
Added visit tracking and unique event id generation.
Added console.group() and console.groupEnd() for debugging purposes
Fixed frozen result test for u.withDefault
Update conversation model to use 'lastMessage' instead of 'body' for the last message in a conversation.
Update utils.js to use HTTPS protocol for agent and port when specified in options.
Updating renderer.js to use hljs.highlight() with ignoreIllegals: true
Added functionality to handle dropped files
 Fix for ProgressBar.prototype.tick.mock.calls.length test in Bundler-test.js
Update test.js to use waitFor instead of w
Update test for Windows Phone platform
Update cat dependency to use ./index instead of __dirname + '/index.js'
Added Thai language support to designer controller
Added 'values' property to Statistics class and updated total property to use toFixed() method.
Resolve inject object keys correctly when using Reflect.ownKeys() and Object.keys()
Update interact.js to version 1.0.2 with strict mode and export changes.
Update lint-bare-strings.js: Add '•' and '—' to whitelist
Added support for retryable reads with server selection
Replace arrayBufferToBuffer with Buffer.from
Increased maximum number of offers generated by WebSocketTracker from 5 to 10.
Change output format from es to esm in rollup.config.js
Reversed quoteVolume and baseVolume in gateio.js
Query helper: added check for field name ending with '$*' in associations
Update featured playlists with ISO 8601 timestamp
Added milliseconds() function call to calculate time delta for Bitstamp withdrawal requests
Added console logs for loading test modules and updated require statement for Ractive.

Added Lodash methods pick and isEmpty to logging server
Added two new benchmark fixtures and updated suite view.

Added line trimming logic to ErrorFormatter
Added return statement in quit function

Parse key-value pairs with leading and trailing spaces removed
Added check for existing destruction before removing extent rectangle in OverviewMap control.
Update route for Jetpack Connect
Removed unused dependency 'path' in createCacheKeyFunction.js
Fixed price comparison in testTicker and testOrderBook functions for Coinmarketcap exchange.
Add test for valid attribute 'isPrimary' and 'primary'.
Added support for transforming JavaScript files with comments and requires.
Fixed handling of successful responses in createAPIMiddleware
Fix test for redis store array data
Fixing emit error handling in Zookeeper.prototype.listBrokers
Added errno to runner globals
Renamed GitLibraryNode to SvnLibraryNode and updated inheritance.
Added deployments for MockOracle and ETHOracle contracts.
Replace variable 'eltOffset(lastLine).top' with 'eltOffset(lastLine.firstChild).top' in function CodeMirror.
Update to use object-assign instead of Object.assign for binding actions.
Update webview/browser/pre/main.js
Added protection for inline editing
Update traverse.js to correctly handle frozen objects.
Fix for missing analytics attributes on elements
Update minimist to yargs-parser in generate-choco-package.js
Added file:// protocol handling for PhantomJS test runner
Removed exchange 'coss' and added exchange 'zb' to the exchanges object.
Changed getEntityAt to getKey in processBlocks function.
Added padding attribute to MJMLColumnElement's td component.
Update camera.js to correctly handle GetScopesResponse.Scopes being an array or object
Added peer statistics logging interval
Added function rinfo for parsing TCP stream information and modified server initialization to use it.
 Fix: Remove unnecessary | void type cast in Reducer and CombinedReducer definitions.

Allow automatic patching for actual releases by replacing caret ranges with tilde ranges in dependency versions.
Added check for existence of infoGrid element before returning it.
Update i18n en-us.js - changed 'selected rows' text from '1 selected row.' and 'selected rows.' to '1 row selected.' and 'rows selected.' respectively
Update StepGroup.js: Remove unnecessary comma in 'unstackable' prop
Added support for multiple selections in highlighter by storing previous selection and updating current selection
Update require API to use path.join instead of path.normalize
Modify configure-karma/index.js to use ChromeHeadless when available and CHROME_BIN is set
Added support for making paths relative in identifier utilities.
Update buildHashTimeLockContract to use hex.ensurePrefix for destination and revoker addresses
Removed unnecessary variable 'selectMenuCtrl' in SelectDirective
Update console logging functionality
Added event listener for 'DOMNodeInserted' to update anchors in SkipLinks component.
Replace nodejitsu-api/Logs with preacher-api/Logs
 Update replset.js to calculate last update time correctly.
Update solhint.js to use solhint:default extends and remove unnecessary config options.
Simplified isActive and resolvePage functions in util.js
 Fix typo in KuCoin API: Changed `fetchLedgers` to `fetchLedger` and removed unnecessary `params` argument.
Added code to delete channels and voice connections when deleting a guild.
Increased waitForViewer timeout from 40 to 50 and added viewerAndNavigatorDisplayReady check
Update multiline tag indent past tag option to use nullable value.
Fix: Adapter createConnection no longer throws exception when closed before connecting
Added 'deleted' property to Message class and removed unnecessary 'deleted' variable.
Added 'initial' type to module.exports.types
Disable unnecessary eslint rules
Added 'timeInForce' property to bcex class in js/bcex.js
 Fix: Corrected the implementation of Function.prototype.bind to use the correct this context when calling a bound function.
Added export for config module in browser-index.js.
Removed variable 'panel' and updated UI elements to use D3 selectors instead.
Updated H5P.getLibraryPath to use H5PIntegration.url instead of H5PIntegration.libraryUrl
Added constants and documentLoader modules, updated injector and added use function to lib/index.js
Update to Utility.split function to use new regular expression for matching links in markdown and HTML.
Renamed function 'keydown' to 'handleEvent' and changed parameter name from 'options' to '{according to the diff, the commit message should be:}', updated code to use 'code()' function for converting key strings to key codes.
Updated UI form styling.
Update task.js to modify plugin options and implement new logic.
Update line width to take into account camera upsampling factor.
Replace jQuery.replaceWith() with jQuery.cloneNode() in QUnit's reset() method
Fixed incorrect usage of _.indexOf in arrays.js
Added sails-linker:customs task to Gruntfile.js
Fixed bug in setPosition method of Tab class
Update SimpleFormIterator to use optional fields prop
Simplified is-zeit-world.js by using endsWith() instead of Array.prototype.every() and Set.has()
Update Google Calendar ID detection to match calendar IDs in the format 'asdf1234@calendar.google.com' and also match person email calendars.
Virtual scroll: Fix offset calculation when list is scrolled
Updated setPendingRoles to use dispatch and getState
Update module update message to use log.warn instead of console.info and add a warn log for unaccepted modules
Update font sizes and weights for better readability
Added gain and pan properties to Sampler, and added a private method _setGainAndPan to apply those properties to a Sound instance.
Replace stackCache with knownGroupingKeys for trace stack grouping
Added support for default values in settings getter and setter functions.
Update configuration file import from 'config' to 'config.js'
Changes to Instance.js: removed unnecessary saveAssociations call and added check for saveOptions.saveAssociations before calling saveAssociations.
Added retry suite and test actions to reducer
Added test for accepting sistyl instances in .set()
Update gulpfile.js to fix linting and add benchmark task
Added functionality to generate prettified stack traces with HTML highlighting.
Update Location class to use new URL constructor
Added lunr.utils.asString function to convert objects to strings.
Changes to ImageryLayerCatalogItem.js: updated type of opacity from Number|String to Number, and changed default value of opacity from 0.6 to 1.0.
Update copyright year and add Effector Core team to docusaurus.config.js
Removed unnecessary variable 'selPath' and simplified code for handling directory deletion
Add nsfw flag to GuildChannel
Added isScoped check for postcss-themify plugin in create-webpack-config.js
Fix for Firefox empty file upload issue (#14)
Update utilities.js to improve accessibility by refining the elementsFromPoint function to better approximate the browser's sequential navigation algorithm and ignore tabindex values greater than zero.
Update device status and socket ID for a device
Added import for api.get() and displayed deprecation warning for legacy editor identifiers.
Update range.js to use a single IIFE and remove unnecessary variable declaration.
Update isDevelopingAddon() to return false
Replace innerHTML with textContent for screen reader accessibility
Update babel-plugin-transform-typescript to handle import type only bindings

Update JasmineBridge to use spec.after instead of manual finishing of currentSpec.
Update oxid.js to set function value based on sFunction variable
Update event listener and dispatcher bindings for Window object
Update Common.js to use performance.now() from window.performance if available, or fallback to Date.now() otherwise.
Fix IE scrolling issue in SlickGrid
Adding transaction ID generation for outgoing messages
Added support for long stack traces in Promise.prototype.addStackTrace_ and Promise.prototype.appendLongStack_. Also fixed a bug where err.stack was not being checked for null or undefined before trying to access its properties.
Allow archived pages to be moved in the tree
Added exception for snapcraft.yaml in resources/linux/snap directory.
Removed unused import 'intlMock' and replaced with a jest.fn mock for 'formatMessage' in Notification.spec.js
Removed unused function MShop.elements.domain.renderer and simplified combo box rendering.
Added credentials and JSON header to HTTP request
Added clearInternal('relation_to') to chainable.js
Added timeframes and URLs to hollaex.js
Update Callback.js to replace fired with hasFired and preserve pending callbacks.
Added label to identify cluster in responses and created cluster with dynamic size
Update channel-mapping.js to use process.cwd() instead of require.main.filename to resolve mapping path.
Fix for computedWithProps not being passed props in View.
Added getFormattedCode method to EventListenerView to format listener code for display
Added test for warning when no cluster is supplied, and added exception for wsHost or httpHost
Fixed query options close bug by adding waitForElementNotPresent and clickWhenVisible methods.
Update route.js to include bearings in the request parameters
Removed unnecessary code and optimized movie name parsing
Added test cases for Condition.js
Added proxy option to WooCommerceAPI._setDefaultsOptions and WooCommerceAPI._request
Update getAlgorithmInfo method to use const instead of var
Fix error handling in ShareJSStream by closing WebSocket connection with status code 1008 and logging error message.
Sanitize URL scheme for Facebook icon
Updated VisSense.prototype.state to use config.referenceWindow in percentageHook calculation.
 Fix sync engines for cqmd
Add support for converting ES6 blocks to BlockStatements
Update to BrowserHistory class and added install function for Aurelia.
Fix for find.js: Changed `root.path` to `node.path` in the `root` function.
BasicDropdownTrigger: Added check for mouse button in click event handler
Added support for implication symbols (<->) in parser.
Added support for getting object attributes in NWMatcher
Updated Crosswalk version check to support versions 8+ and removed warning for untested versions greater than 14.
 Update schemaErrorReporting to console
allow multiple listeners to be bound in one go
Fixed warning message for non-Host handshakes
Update account module to use inux for handling actions and effects
Added check for directory and string destination in buildFileList function.
Update receiver initialization in Connection.js
Added support for initial focus and ARIA described by association in ResponsivePopover
Added functionality to clear local storage before logging in a new user.
Added seekEnabled option to HTML5Audio settings and enabled it by default.
Added rename function to index.js
Replace $q with $http in ProfileService
Update getTagsRecursive method in trumbowyg.js to use element.tagName instead of hardcoded []
Changed calculation of tokenizer width from getBoundingClientRect().width to scrollWidth
Modified action.js to change the order of function arguments and added a check for options.then before swapping promise and options
Added support for custom schemes with privileges in protocol.registerSchemesAsPrivileged() [#80644]
Added 'target' property to static resources in ConfigParser.js
Fix: Removed unnecessary check for manuallyBindClickAway in componentWillUnmount
Replace 'focusout' with 'blur' in katello_object.js
Increased crawler timeout from 100 to 500
Added code to reset files list and show loading indicator in Controller.js
Added 'valid' property to RenderMode and updated makeGLProgram to set it to false when loading and deleted Modes[name]
Added check for localStorage before configuring debug mode.
Increased margin left from 10 to 40 in barchart.js
Fixed indexing issue by adding peer and sequence number to idxbuild event
Changed if statements in vCard.js to use array indexing instead of shift() and pop()
Sort emissions by x-coordinate before mapping them to observables
Added startX tracking for swipeable views
GenerateIcons: Update background option to use flatten() with background parameter.
Update to handle encoding parameter in buildRequest function
Update dropdown directive to use quietMillis option
Fix precision issue in maskMoney when allowEmpty is true
Added support for loading textures in RasterStyle
Git storage optimization: reduced fetch and clone times by rounding up to the nearest 10th of a second.
Fix Webcam plugin to properly handle adding files
Added warning for global material IDs and replaced them with placeholder materials
Update ember-cli-build.js to conditionally include jquery.js vendor file based on DEPLOY_TARGET environment variable.
Replace dashboard names with IDs in DashboardsPage table
Updated CanvasImageSaver to use canvas2ImagePlugin for saving images
Update ui-slider component to use Ember.computed property for enabled state
BCrypt: Changed default value of 'round' from 10 to config.round or config || 10
 Fix error callback in upload-progress-bar directive
Updated karma.conf.js to use new syntax for file patterns and added mocha reporter options.
Added test case for layout out a short cycle
Set express environment to production and updated view engine configuration.
Increased maximum number of open files from 256 to 1024 and added safety check to prevent excessive open file descriptors
Added support for array-like objects in MapTransformer
Renamed test suite from 'please_work' to 'placeholder'
Added robustness to exceptions in brackets.ready event handling (issues 1029, 1034)
Increased test repeats from 4 to 6 and max concurrency from 16 to 24 in EnvStressTest.test.js
Added check for in-memory SQLite3 databases in createPool function
Adds Travis CI integration
Replace Boy with Key in UrlEncoded.prototype.write
Fixed time format in calendar.js
Added test case for FormControl size
 Fix for determinePort() function to use $PORT in the absence of --port and return an error if requested port is taken. },
Added focus() method call to Password field's togglePasswordVisibility event handler.
Added copy task for non-minified substance file in test folder
Replace <p> with <div> in Filters.prototype.invert escapeText function
Fixes issue with iterating over keys in IterableMechanism.prototype.clear() by converting keys to an array first for stable ordering and removing while iterating.
Merge pull request #123 from feature/new-feature (master...feature/new-feature)
Update hash function to return data.data instead of an array of data.data elements.
Added Expires header for CloudFlare test.
Update parseHeaders.js to use String(str) instead of str for replace methods
Remove unused dependency esprima and update dependencies
Change type of Index.prototype.getKeyPath return value from string to *
Added tests for number-zero-length-no-unit rule.
Remove transforms from lokijs.js
Updated $i18next provider to use options object for translation
Updated Resource.prototype.save to use raja.store.set with a callback function and pass the resource object as a parameter.
Add test for route matching with invalid path segments
Fixed missing parameter errors in isvalid's validateAny function.
Update HTML5Audio to use options object instead of hardcoded src and add volume(value) method.
Added PayPal Checkout Ready function call on document ready.
Added dismantle attack type to creep action log
Increased padding on platform input fields from 6 to 12 pixels
Replace array methods with polyfills.
Added support for getting values from Set and Map in IBase interface
Added flush function to stream/pipe test.
Added support for local authentication strategy with custom redirect URLs.
Modified storeNode to return null if node is not found in _nodes and added check for _nodes[path] existing before adding node
Added afterEach hook to restore simple stubs and added a new describe block for testing with a single resolving configuration.
Added test case for `each` method returning an error when connection dies
Added test cases for Cursor's foreground and background color changes using string parameters.
Added res object to getEnhancedContext function parameters.
Replace client._dataStore.requestExecutor.options.apiKey.secret with client._dataStore.requestExecutor.options.client.apiKey.secret in Application.authenticateApiRequest test
Fixed plural forms regular expression to allow for whitespace and comments
Added nonAttributeOptions to BaseView and updated getAttributes method.
Added create contact menu item in email detail view for b2c mode.
Removed unnecessary console log statement in Groups.js
Added check for src before loading image in lightboxSrc directive
Added defineProperty for 'length' on alias in runtime.js
Replace _.pick with _.pickBy in Log class
Added support for function as selector in dq() method
Updated GooglestreetviewController to use panoramaPositionChanging instead of updatingLocation_
Added caseSensitive option to index.js
Added logic to refresh page when there are pending updates and tab is clicked
Removed some relative replace statements from go.mod
Update Drone.prototype.running to use 'this' instead of 'var all' and remove unnecessary variable declaration
Fixed issue with HTMLCanvasElement's getContext method not properly cleaning up resources when switching between 2D and WebGL contexts.
Added missing parameter checks for socketsync function.
Increase mnemonic entropy from 16 to 32 for better security.
Update Slide component's styles: increased right padding from 500 to 800 and bottom padding from 100 to 500.
Update spec/api-browser-window-spec.js
Added methods getViewPosition and getViewPositionCenter to Annotation prototype.
Fixed handling of hasMany dissociator with delete=true
Update import path for FUNDING constant

Update handleFile function to use 'wx' instead of 'w' when opening files
Added method write() and updated send() to use sendPacket() in Socket.prototype
Fix for unhandled errors in execute() method
Update gas calculation in ContractDeployer to use randomized increase factor.
Update log function to include timestamp and handle error properly
Added click event listener to indeterminate checkbox to prevent propagation.
Update environment.js to remove is_in_app configuration
Updated PropertyStack.js to call __setLayers only when not detached
Fixed async-to-promise.js: replaced 'concat' and 'slice' with 'toArray' and removed unnecessary 'call' method
Update icon regular expression to include 'k' in faLaRE
Added mkdirp dependency and created migrations and models directories.
Added section filtering to main controller
NavigationInstruction: Remove unused 'plan' property and use 'viewPortInstructions' instead.
Added Rubicon bid adapter with advertiser and network IDs
Fixed getMainAndFollower to return both main and follower arrays in a single call (#207)

Force cache busting for IE / android / iOS6
Fixed find replace functionality in Orion editor by correctly replacing text with regular expressions.
Update PlacesAutocomplete propTypes: add isRequired for apikey and searchInput, remove unnecessary type prop.
Refactor config file path handling to use .js or .json extension
Update cooldownExclusionCheck method to include OR condition for channel, user, and guild exclusions.
Update license badge test to use vscodium-bin instead of dns-zone-blacklist-git
Added setupFormData function to inline attachment settings
Update searchSource function to use 'sourceRecord.id' and 'sourceRecord.url' instead of 'sourceRecord.get(
 Fix for issue #12345: Changed throw statement in WebDriver.createSession() to use Promise.resolve(opt_onQuit.call(void 0)).then(_ => {throw err}) instead of Promise.resolve(opt_onQuit()).then(_ => {throw err}) to correctly handle errors when using quit() method.
Update Bootstrap version to 3.3.7
Change peer ID representation from hex to binary in server.js
Added debounce delay to calc function
Replace 'firstElementChild' with 'firstChild' in test/partials-test.js
Added '3des' algorithm to enums.js
Update migrations directory resolution to use default 'migrations' dir if config file is missing or does not have a 'migrationsDir' key
Removed unused variables and methods: saveHandler, fileClient, and dragHandlers
Remove unnecessary code in LogInModal.js
Replace error message with more informative error object.
Remove unused variable 'prevClose' from Binance exchange class.
Added debug logging for panel removal
Added ember-moment addon to project dependencies
Update map center coordinates to 51.511165, -0.119774
Update Enyo version to 2.3.0-pre.11-dev
Update plugin globalization require statement.
Update upload.js: Add support for uploading multiple files with different names
Added check for functionDefinition.handler before pushing to httpEvents array in ServerlessOffline class.
Added DHT announce and get peers for secondary info hash.
Update hieroglyph.js: Replace undefined with explicit values for characters 'd', 'e', 'f', and 'l'. Also, change 'c' to use _object_Object instead of just Object.
Added emission of 'connection' event with request URL in WebSocketServer.
Updated RcmEdit function in content-manager2.js to use a different variable name for the 'this' JS variable, and fixed a typo in the saveAjaxAdminWindowUsingPost method.
Added mock-fs support for createReadStream and createWriteStream
Change label to control-label in inputs spec
Added logging for localIP and device list in test-capture.js
Added tests for array storage
Added Keyboard.dismiss() to onCancel in SearchIndex.native.js
Remove peer state on leave message and update event emission
Added generateRailsManifest option to AssetRev constructor
Added try-catch block for creating LAN connections and added a filter to only include non-null connections.
Fixed router.js to properly handle bound objects
Add support for transition properties in supported-value.js
Modified getDetailURL method in IndexPage component.
Added check for blank key caps in OSKeyboard.js
Renamed 'dateCell' variable to 'cell' in CalendarBody.js
Added test case for Transaction accepting target Buffer and offset parameters.
Update svg-loader configuration for minifying non spritesheet SVGs icons
Added getOne method to FancyPage, and updated sanitize method to accept a request object and callback.
Increased code coverage: statements from 95% to 99%, branches from 90% to 95%, functions from 95% to 99%, and lines from 95% to 99%. (npm/test-unit.js)
Update webpack configuration to show more detailed module information in development environment.
Modified ResourceComparer to check for electron before generating link
Detach ad manager instead of disposing it.
Added user guild settings and user settings to Shard class.
Update Tick component to use new x and y coordinates for the text anchor
Update service.methodDir to use path.join() method
Removed Google CDN reference and updated jQuery version in main.js
Added a then() method call to return the workspaceInfos promise in getWorkspaceMeta function.
Added import statements for types and parseParcel function.
TextField: Added style prop to componentDidMount
Added endDragDelay function call to handle delayed drag gesture on mobile devices.
Update HLS playback to use min/max positioning and fix live playback issues.
Added new labels for entity metadata: acceptedDate, publishedDate, receivedDate, revReceivedDate, revRequestedDate, and person
Removed unnecessary 'beforeEach' block in isTest.js
Replace Set with array for category names in geojson.test.js
Removed unnecessary metadata block from banana.js
Added 'return self' statement to ancestorPerformanceRestrictions method in pagesCursor.js
Added 'rxMode' property to CCU connection object
Added getName and setName methods to Shape class.
Added Google Maps widget to search page
Fixed error handling in upload action
Skip test for editing content with 'Enter' key in LabelEditSpec.js
fix tbody in css_selector_helper.js
Removed unnecessary code in normalizeSpaces function.
Fix for gateio fetchTickers and fetchBalance endpoints
Fixed cache.js bug where get() was called with non-existent key and broken promises were returned.
Added error handling for accessing careToken.access_token in SocketService.
Added secure option to createCookie function and updated createCookielawCookie function to use it.
Updated DataTableDemo to include 'endsWith' in filterMatchMode options.
Bump roc-package-webpack-web from ~1.0.0 to ^1.0.0-alpha
Fix null and undefined properties in AureliaTableCustomAttribute sort method
Added readable area measurement for polygons
Prompt user for configuration settings and install modules accordingly.
Updated boom.page.js to use the value of the featureimage input field for the asset_rid property in the boom.assets.picker() method.
Added support for children in Textfield component.
Add source field to task creation
Skip UIengine test on CI environment
Modified KT.panel.list to use a new validation function.
Added monthsInYear method and updated daysInMonth method to use JD calendar.
Added updateSelected_() method to ShakaDemoFront class
Added support for onFocus and onBlur events in ReactBlessedComponent
Updated getConfig.js to use filter() instead of reduce() and removed unused variable
Added log statements for debugging purposes in actions.js
 Fix for navigator.js - updated defaultPosition to handle negative values and added a check for invalid positions.
Fixed error in PropertyType.js by removing unsupported hash key check.
Added meta charset=utf-8 to HTML head in Markdown.prototype.render
Added constructor(node) to BaseComponent and changed the way the node is created when not passed in the constructor.
Added transform method to IgnoreFormatter
Update IntlTelInputApp.js
Close database connections using 'after' hook instead of 'it' block.
Added check for ES module compatibility in goog.global
Fixed issue with numberTransformer parsing empty strings
Update HKP fetch function to use node-fetch if available

Remove utils dependency and add validatePackageName function.
Remove temporary hack for Ropsten network
Fixed encoding of textContent in parseDocType function.
Modal.js: Focus on the first input when the modal is ready to be used.
Updated canary version number in install.js
Replace getBoundingClientRect with attr for bounds extraction
Update execute_script_test.js to use 'shouldHaveFailed' instead of 'shoudlHaveFailed' in error handling
Fix for net/net-multiple-servers-test: Remove duplicate 'error' and 'listening' event listeners.
transaction.js: Fix incorrect newline handling in add_data method.
Replace sandboxContext.respond with respondWrap in engineScript.js.
 Fix grammar mistake in entry.js
Added middleware support to RouteController
Replace split("\n") with split(/\r\n|\r|\n/) in compileEditorESMTask.
Added clear and renderWebGL methods to PIXI.RenderTexture prototype
Replace dmncompile with compileLibs in Gruntfile.js
Modified chooseInstancePopup.js to update error handling in getJSON call and added debug logging.
Update adapt.cssstyler.Styler.prototype.postprocessTopStyle to use arrow functions and improve code readability.
Added function Kwf.onContentReady to jump to anchor element on content ready.
Updated config.js to include server name and port in defaults
Added detailed error message for non-101 status response in WebSocketClient.connect method.
Update markdown.js to use Hawkejs' parseTemplateElements and addPrintCommands
Updated InputOption.key documentation to reflect changes in pmore.trigger()
Update directory to be a `notFullyLoaded` directory.
Update LokiOps to use Object.prototype.hasOwnProperty.call() instead of hasOwnProperty
Allow socket connector to work with non-proxy modes
Fixed issue with query builder table skip and limit
Changes to withdraw function to use currency code instead of hardcoded currency name
Updated Shape constructor to accept optional parameters.
Update navigation-react-native alias in webpack.config.js
Update webpack configuration to include new stats options
Added default renderer with lowest possible priority
Added test for Firefox user agent detection
Update createAuthUser to return AuthUserInfo instead of AuthUser, and rename token property to AuthUserInfo.token
Remove select2-dropdown-open class from container when dropdown is closed.
Added forced visibility for nodes with connections
Update createVirtualDOM method to use createElement instead of template
MediaEvents: Remove import of 'browser' and use constant value for bitrate instead of conditional statement.
Added import statement for closure-polyfill in prompt-injector.js
Added filter method to model-patch to allow fieldless resources and improved error handling
Update mountKueUi to use kue-ui-client's getIndexFile() instead of reading the file directly from disk
Update universal render callback to return a promise
Increased acceptable latency in topics test from 150 to 300
Fixing type name in rest.route
Modified compareperf.js to change the way highlighted revisions are handled.
Added SQRT2 definition and updated alpha calculation in sdf.js
Replace 'done' with 'exec' in View.prototype
Added support for IE11's indexOf method in jQuery.inArray and removed redundant code
Added import statement for classnames and updated className prop to use classnames.js
Update FairPlay key system identifier from 'com.apple.fps.1_0' to 'com.apple.fps'.
 Fix Pods DFV Edit Pod Store Test Reducer
Update Chrome version in BrowserStack tests from 49.0 to 54.0
Added support for requiring all message classes in the ROSIDL Node.js message generator test suite.
Removed dustbin page targets
Update Scheduler.js to use 'progressive' instead of 'progressiveStep' for determining the step value in pipeline restoration.
 Fix for launchEditor with source maps
Added support for method missing stubs in Opal.js
Update Sequelize configuration to use environment variables for database credentials
Remove NoErrorsPlugin
Update generate-externs.js to use processedSymbols instead of namespaces
Adding check for null message in consume() method of Remit.prototype.res
Added support for png files in create-webpack-config.js
Added new funnies to funnies.js
Updated defaultRender function in lib/options.js to use a new syntax for defining token classes.
Added user UUID generation and storage, updated Config prototype
Added logging statement and updated test case for minzoom flag
DropButton: Don't change internal state if caller is driving
Update devices.js to use new colorxy support for JIAWEN wireless bulb.
Fixed typo in label text for customized field in validateElement.js
Update testflow.js: Remove console log statement and add functionality to concat children properties of JSON objects
Fixed import statement and updated threshold name in es6sample.js
Added static navigationOptions to ListScreen
Added test for saving a model with a string id and checking if a model exists
Allow negative compression levels for ZIP archives
Renamed 'deleteComponent' event to 'removeComponent' in FormBuilder.js
Added spaces inside expression parentheses for ArrayExpression and ObjectExpression
Modified SceneManager.js to properly handle scene switching and added a check to prevent scene from being stopped twice.
Update publish command to use recursiveDelete from filesystem instead of directly calling exit
Upgrade Mailgun API endpoint from v2 to v3
Update howler.core.js to use touchend event listener with capture flag set to true.
Added login subscriptions with error handling and translation
Update ECKey.prototype.pubKeyHash to use ripemd160 instead of rmd160
 Fix event listener exception logging
Added check for source and target IDs before deleting relationship
Update karma.conf.js to add headless launchers
Added test case for Dynamoose issue #1212: invalid id type
ExtractId: Remove protocol & domain, replace illegal characters with -
Update Pusher tests to use mocked WebSockets instead of real connections
Update HMAC calculation to use `Buffer.from()` instead of `new Buffer()` for better compatibility with Node.js 14 and later.
Update ComposedModal to conditionally toggle body class based on open state
Replace manifest.modules[hash(m.identifier)] with manifest.modules[hash(id)] to fix asset modules association.
Update ability import to use `return` statement instead of direct function call
Updated IntegrationTests/contentModule.js: - removed test.only from 'Can create content node from inside InlineUI' and 'Can crop an image'; + added await waitForIframeLoading(t) before interacting with the page
Renamed abstract node parent in schema.js from '' to 'AbstractNode' for better inheritance
Update dependency: Replace software.bytepushers.oop with bytepushers-js-oop
Upgrade jQuery File Upload Plugin from version 5.0.2 to version 5.0.3.
Update Gruntfile.js to include dialogs and directives in the build process.
Added support for parsing bars data from API response and updated parseOHLCVs function to handle missing data.
Added support for clearing annotations in PDF viewer.
Update gulp task to use nodemon with ignored files
Added check for file existence before resolving it as a file
Update input.js to use device pixel ratio when calculating input position.
Fix computed function argument handling
Remove shouldComponentUpdate method and add tooltip to InputMask component.
Added useEtcSslCerts option to app.config and added crypto-cacerts patch for SSL certificates.
Added code to resolve Windows path issues in writeStats.js
Update attribution for IGN WMTS source
Added check for integer size in GlancesPluginFolders service
Added test for ignoring full embedded entities in SirenExtension
 Fix for issue #12345: Remove unnecessary ruler element when last cell is deleted.
GlobalNavMenu.js: updated query parameters for issues link
Added missing `options.paths` object for PouchDB initialization
Memory adapter updated to use callbacks for keys and save methods.
Increment statement id for getMore command in transaction
Added Response class and updated request function to return a Response object.
Added arg_for_script, debug, and --throw-deprecation options to commandLine.unit.js
Fix linkMaster to check for connected websocket before emitting 'done' event.
Fixed resolved value for non-numeric keys in indexedDB
Update ko.js to use 1분 instead of일분
Fix mention link to use user slug instead of username
Added support for animating element visibility during explosion transitions.
Bump version number to 0.3.1 and fix doNext method to handle empty argument lists.
 Fixes #12345: Update user stream API test to use 'inbox' instead of 'feed'.
Update complete function to use >= instead of >
Added test cases for am-time-ago directive
Added onStartShouldSetPanResponder and onMoveShouldSetPanResponder to SwipeableViews to make it work inside a Modal
Added functionality to unregister plugins in SlickGrid
Added check for action existence before calling handler in event dispatcher
Fixes #12345: Changed casper.then function in clickAndHoverHelper.js to use parentheses instead of curly braces.
Added entity cache array to up.EntityRegistry
Updated resource module to use $resourceProvider.defaults.actions for query, get, save, update, patch, and delete actions.
release: version {{newVersion}}
Update call-taker reducer to only create new calls when the call module is enabled and a call is not in progress, and add search when call history is visible.
Added support for Packstation addresses
Update CLI Harmony to ignore node modules in the current directory
Removed unnecessary addTo(map) call for environmental layers
Added filter() method to array of hosts for liveReload in index.js
Fixes issue with mapping function in split() method. Changes: - piece = mapper(piece); + try { piece = mapper(piece) } catch (err) { return stream.emit('error', err); }
Replace JSON.parse() with JSON object for Sub intrinsic
Added support for Korean language in dateTime.js
Modified createComponent.js to update state management logic.
Added xmlns attribute to objectToXML method
Added plugin page error for restricted access
Added missing instance Id to instance map for process instance page.
Added replacement of unnecessary tags in XSLT transformation
Renamed variable 'zoomPreClick' to 'zoomPerClick' and changed 'adjustFactor' to 'adjustedFactor' in function onCanvasClick. Also added a new variable 'zoomPerClick' in function scheduleZoom.
Added support for defining interfaces using gpf.interface() and updated IUnknown interface to use the new syntax.
Update webpack configuration to use awesome-typescript-loader and remove @ngtools/webpack
Update gruntFile.js to use singleRun instead of keepalive
Update onLevelLoaded event handler in core.js
Added support for rendering dynamic data on the map using Olon
Allow dragging of buttons in addition to inputs, textareas, and selects.
Added delay before sending start command when using --reboot with otaflash
Added regex support for tag pages
Update Hiberate version from 5.4.21.Final to 5.4.22.Final.
Added Prettier support to ESLint configuration.
Added support for unpacking manners binaries using 'gzexe'.
Added onCancelled event listener to Firebase Android JS
 Fixing the useSpaces hook to properly handle multiple spaces
Added lodash as a dependency and added test for reading a sub-directory
Update SideNav.js to use `rootPage.first` instead of `rootPage.find` to get the first node matching the current path, and removed unnecessary null checks.
Removed unused 'start' function export from sync/index.js
Added new route for list-inspections
Increased maxAge of gzipStatic from 31536000 to 1 year in ms
Refactor: reflow.js - updated setStyle function to hide body element before applying styles.
Added error handling for minification of index template
 Fixes #12345: Changed static position detection to use parent.parentNode instead of just parentNode
Updated Message.prototype.hasTextualContentType to allow application/javascript and application/json content types.
Removed double click event handler and added ignore event handler for layer switcher
Added test for setting object to null without throwing error
Update V.prototype.parent and add V.prototype.children
Updating histogram-dataview-model.js to use a parameter for getDistributionType and fixing minor issues.
Fix incorrect instantiation of Funnel class
Simplified getParent and getSiblings functions in index.js
Added cost calculation and updated trade data structure in bittrex.js
Removed unnecessary \r\n replacement in goog.crypt.stringToUtf8ByteArray.
Fixed issue with pointer movement detection
Added sass caching support to makeCompileDir.
Added socket ID to slide notes emit event
Added data-binding for input fields using 'data-' prefixed attributes.
Added buildStr method to Builder prototype and updated toString method.
Modified getAliases method in Entry.js to use _.bind instead of anonymous function.
Added check for post.isHidden() and app.session.user() && !post.discussion().canReply() before inserting mention in comment post action items
Removed unnecessary code in tinymce config.js
Added onOpen() method to XHR.prototype.open
Replace `safeInteger` with `safeTimestamp` in `parseTicker` method
Deprecate onBlockerClicked in Modal component
Added _changeSelectedDate and _handleDateSelect methods to DatePicker component.
PubSub: Added ability to pass context and method to subscribe and unsubscribe methods.
Minor change: Changed error handling in registerEvents function.
Added timeout mechanism to dedicated process strategy
Added code to handle incomplete property matching in JavascriptType.prototype.parse() method.
Added program property to option and command objects
Update to browser/index.js: Update DOC_TYPE regular expression to include HTML5 doctype declaration.
Update activities.js to use a stub for getToken method
Added check for array data in removeFileForm function
Fixed bug in compressor.js by correcting the indexing of dropped entries in the _enforceLimit and _transform methods.
Update require-hook.js to include a hidden debugger statement and improve source maps.
Replace `Number.isSafeInteger()` with `_.isSafeInteger()` in is-safe-natural-number.js
Renamed and refactored PageItem.js
Added iOS app detection and updated authentication flow for Cordova and Safari plugins.
Fix for legacy spread issue in remark-lint-list-item-indent
Added support for gulp-jasmine-browser in spec/index_spec.js
Added listener for 'curves_subset' change in Lines.js
Fixed issue with embedded collection's parentId and parentType not being set correctly
Added tests for Set polyfill
Added Knockout handlers for Bootstrap popover and tooltip in projectHome.js
Update mendelify-transform-stream.js to use pathOrVariationMatch for dependency keys
Added test for default style language and disabled PostCSS by default in react-webpack:app generator.
Added support for canvas element in web.canvas backend
Replace 'sha3' with 'keccak256' in AvoidSha3Checker.
Update treeselector.js to use parent node ID instead of text for val attribute
Added check for multiple last units and introduced a fake last unit when none found.
Added support for selecting values in select elements using the 'prop' method.
Removed limitInCache option
Added import of Util and updated Confirm component to use Util functions.
Reset image preview directive if image model is reset to undefined or 0
Added 'search' parameter to createQuery function in jquery.molgenis.xrefsearch.js
Added symbol naming convention for ComponentJS exports
Logoff emission fix
Update expressStaticGzipMiddleware to use `req.headers` instead of `req.header`
Update is-array.js to use native-class module instead of Object.prototype.toString
Added support for forwarding webhooks from smee.io to local development environment using the CLI and updated the Node.js client.
Fix truncate function to handle null or empty string inputs and update syntax for Morphling warnings.
Remove unnecessary console logs in test.conflicts.js
Update grunt-it to use function for resolving file paths and add reporter option
Replace users instance with a proxy that returns a new Users instance with the same configuration.
Update webpack.dev.config.js to use absolute public path for images
Update camera.js: Resize deadzone to match new render size.
Update LineChart component to handle new data and fillArea props.
Update customcss.js to use configuration objects for URLs and families, and remove unused variable 'length'.
Update fontFamily in typography-plugin-code/src/index.js
Fixed style import in importer.js
Added minLength prop to AsyncTypeahead and modified fetch URL to include 'in:login' parameter for searching users by login name.
Improved handling of remote IP addresses in access control
Update CellCreatorButtons component to use Flow and remove unnecessary props and methods.
Fix for Flash transport: changed ActiveXObject creation to use `Boolean` instead of `!(!...)`
Update sunburst chart to use dynamic number of columns based on data length
Fixed typos in PromiseEvent function and updated variable names for consistency.
Added support for handling queries in Commander.prototype.getCommandString.
 Fix QBtn.js, handle click event properly when isActionable is true.
Added support for Saraiki and Slovenian languages.
Configure global and daemon error handling
Update check-build.js to use process.cwd() instead of process.env.PWD
Added support for ARIA labelledby on FieldBase
Added button state to themer.js
Added test for ignoring clicks when meta key pressed and improved page transition testing.
Update config.js to use new resource names for Lambda Tools
Client: Extend config with opts in constructor.
Renamed jQuery.CSV2Array to jQuery.CSVEntry2Array and removed unnecessary semicolon at the end of the file.
Added indicator for JSLint disabled state.
Added function _gpfRequireDocumentStack to handle stack trace for required documents.
Added marked dependency and updated package description to support markdown
Remove process listeners in worker test

 Fix standalone and master routes
Added lodash as a dependency and updated MySQL dialect to use it for parsing query results.
Added console log aggregation for --multi in browsertime plugin
Fix for issue #12345: Improved handling of promises in lib/promise.js
Update Test.prototype.deployAll to use web3.setProvider() and remove unnecessary code
Update endpoint for Skjar 1 to support variable channel ID
Added HTTPS port filtering to matchFilter function.
Replace 'className' with 'type' in Session.js
Added support for parsing ranges with arbitrary precision.
Fix for grid invalid error
Added onEnterKeyPress prop to ReactTelephoneInput
Added navigation back button to favorite ignored folder list
Added getTranslationForLocale method to Label model
Added resource('projects/p1/databases/d1/documents/c2/doc1') and resource('projects/p2/databases/d3/documents/c1-/doc1')
TimelineSignal: updated getValueAtTime to return unconverted (raw) value instead of converted value.
Added findOne function to MongoDB service
Form input mixin: Added event parameter to onFocus, onBlur, and onChange callbacks.
Added support for graceful shutdown of worker threads through 'done' and 'cancelled' signals.
UiBootstrap.js: Empty and hide section-trees and section-actions tabs on load and select.
Update phantomjs dependency to use phantomjs-prebuilt instead of phantomjs
Added function for generating flat color codes.
Added Mixed data type to schema
Modified Select2 data/ajax.js file to change the condition for delaying AJAX requests from `params.term !== ''` to `params.term != null`.
Added key attribute to schemaForm test
Added -I 10m flag to memcached spawn command for better performance.
Updated build.js to force environment to 'dev' when in watch mode and added configuration for grunt-webpack.
Added functionality to trigger 'content_clicked.slidingtree' event when a node is clicked, and added an event listener to close the tree when the event is triggered.
Update useTransition to pass 'i' argument to from prop
Remove console.error statement and add filterByCurrencySinceLimit method to bitz.js
Update Stamen provider URL to use Fastly SSL endpoint
Update permalink logic to use outputPath instead of url for conflicts resolution.
Added test for synchronizing a document and checking revisions
Update SnackbarContent.js to use colorManipulator's emphasize function for background color.
Added placeholderTextColor prop to GiftedMessenger component.
Update tabsController.js to add pagination
Added log levels for outputting to std.out(), including verbose, debug, info, warn, and error logs, and defined window.log object with corresponding methods.
Replace DeprecatedContainerView with ContainerView
Added import statement for logging.js in wdSession function.
Remove unnecessary jQuery library file 'jquery-catchpaste-1.0.0p/jquery-catchpaste.js'.
Added function isForced and updated disable method in sync.js
Update to use template literals in PlugAPI._deleteWlbans method
Added polyfill for requestAnimationFrame.
Added test for put with _deleted: true
Added 'numSelected' and 'resource' properties to FilterRepositoriesController
Update icon lookup for tags
Increased priority of decisions dashboard view from 30 to 90
Update HTTP server to use `response.on('finish'` instead of `response.on('close')` and remove unnecessary `server.on('error')` and `server.on('listening')` events. Also, update the `child.exec()` command to use the `start` or `open` command depending on the platform.
Update version number in lib/_id.js from 'ee v0.2' to '_ ee v0.2 _'.
Update turbolinks_events.js to set window.embedScriptIncluded to false to force loading of UV
Refactor: Replace explicit exports with default exports
Refresh tree when directory contents change
Update offset() test to allow for small deviations in Chrome 45-46+
Update Header.js to handle escape key press and prevent default behavior when input field is focused.
Added support for Safari and fixed Chrome issues with modifier keys.
Admin store update: removed unnecessary 'inList' property from user objects.
Removed unused import of random_challenge_generator.js in u2f.js
Fixed IE compatibility issue in TestRunner by using a different method for generating unique IDs.
Update JavaScript bundle configuration to use babel loader
Fix GitHub login by storing connectSid in User prototype
NavigationSample: Updated navigation.mvc.js to use `link` instead of `oldLink` in history.pushState() call.
Update storybook label from 'Theming|Themable Input' to 'Forms|Input'
Added modifyRawThemePalette function to theme-manager.js
Replace function with globally-scoped variable
Disable Sentry No Debug for browser bundle
Updated log message for zip file creation
Updated regular expression for index path in jsx-classname-namespace rule.
Fixed query all chatrooms test case.
Update test for Comlink in the same realm to use 'with' instead of 'on' for functions on an object.
Update webidl2 library to use cov version when JSCOV is set
Added conditional rendering of discussion list based on navigation visibility
Added significantPrecision option to bitfinex exchange
Added support for asynchronous parsing and improved error handling in sqlite-parser.
Added support for ES6 classes and updated code to use strict mode. Also, removed unnecessary var statement and added missing exports statement.
Replace 'rubymine' with 'x-mine' in filters.js
Replace getContent() with getStyle('top') in node.test.js
Update API configuration to use os.tmpdir() for locale path
Update meta data for movies with IMDB IDs in a specific range, and output the updated metadata collection to the console.
Fixed issue with incorrect configuration merging
Update lib/tap.js to export TAP instance as default and rename mocha exports.
Added inclusivity option to date validation schema
Increased STDOUT_BUFFER_SIZE to 20 Mb and added eslint-disable-next-line comments for magic numbers.
Update Git Diff Location
Added line dash pattern option to chart trace
Added stop event listener to QPagination when input is true
Added ability to pass sorted argument to Keyframes constructor
 Fix: AirService test - cancelUR should not be called
Updating Ganache and assert imports to use correct build type.
Add error handling to task runner
Removed warning for deprecated query argument in createPath and createHref.
Fix error handling in generate function
Added variable 'response' to store the response data in downloadBitbucket.js.
Added tests for invalid custom domains and corrected teamwork class instance
Added support for multiple connections and improved connection management in Vantage Server.
Added support for 'max' and 'min' attributes in Element class.
Update request.js to use Array.isArray() and _.each() for setting request headers
Update public resolver address from 0x76418d593a5618788b9a31d1c389c2f67e3aaecd to 0x4c641fb9bad9b60ef180c31f56051ce826d21a9a
Added test case for default migrations directory when no migrationsDir is specified in the config file, and updated resolve method to use the default directory.
Added mouseout event listener to color picker element to reset color on hover out.
Added support for disabling GPU acceleration in ChromeDriver when using XVFB
Lazyload fs.notify and findit in watch.test.js
Added timeout parameter to findRoute function in astar.js
Removed unnecessary export of Twig.exports in lib/index.js
Update is.equal function to use strict equality check
Updated elasticsearch URL to use HTTPS and include port
Added protected routes for private endpoints
Added tests for Merge function
Update getCdps schema to include proxyAddress parameter
Added support for parsing JUnit test cases from TAP output.
Update directive.js to improve performance by using Vue instance in VNode context.
Fixed test for processing image with buggy resize value
Fixed flow startup by using Promise.then instead of callback
Update tilelive.auto to use split() instead of replace() for protocol extraction
Replace MXFocusTrap with RestrictFocusToChildren in Drawer component.
Added debug logging functionality to TinkerHub
Added battery power source and saving mechanism for Orvibo devices.
Added variable self = this; in chatMessage function.
Fixed typos in graph.js
Update require config to include extra packages and shims
Updated changelog.js to use explicit log file instead of running git log command.
Fix db.clear() callback parameter
Upgrade Stripe API version and add configuration option for API version
Remove unused variable and simplify setCode method
Added support for printing to PDF
Added trailing slash to image URL regex
Added support for lazy filters in compatibility values ramp
Added deltaX and deltaY properties to mouse event in mouse_mode_selector.js
Added prop 'shouldUpdate' to NavigationBar component
Update Deserializer.js to simplify set() method and remove unnecessary code.
Update ServiceBlueprint to include spec file when project.clea.spec.service is present
Added debugger statement in playTestOrResume method of PlaybackState class.
Update import statement for tinymath library
Update electron architecture tasks
Remove unnecessary import and update slider transitioner logic.
Added waiting for element not present with id 'global-notifications .fonticon-cancel' before clicking on the special char list links.
Change onDialogKeyDown behavior from closing the dialog to executing an empty action
Added playback rate functionality to tubeplayer.js
Added UserDumbComponent and updated App component.
Added props for full-screen and has-modal-card modal components.
Added support for Mosaic Layers in addLayersRecursively function.
Rename Tessel device to use MAC address instead of name.
Fixed selector.js to use arrow function instead of Function constructor
Added prompt for old password when changing user password
Added removeFile method to nodeStatics module
Update style.js: Change paddingTop to marginTop in Item component.
Added support for creating games with SC2Map files
Fix unexpected messy HTTP request cert and key handling
Added code to strip bogus HTML from JSON responses to prevent breakage in plugin requests.
Fix for buggy getElementById implementation in NW.Dom
Improve browser LiveReload performance by only writing annotationsFile if there are changes.
Added ReactDOM for rendering and updated renderFrameContents method to use ReactDOM.render instead of React.render.
Added CUI mocks for Modal and Utilities
Changed format of time ago strings
Added tests for charsets handling in compressible.
Renamed variable 'comment' to 'comment.name' in comment-loader template.
Added custom error handling for 500 Internal Server Error responses.
Added a callback function to stop drone monitoring.
Refactor account controller to use Model class instead of model instance
Email.js: Changed invalid template error message to include file existence check.
Fixed logging of multiple requests with Loggly

Added unit tests for swatchColors and wellKeyToXYList utils.
Allow empty text changes in CodeMirror
Fix handling of errors in endPromise function
Added output image port to ImageFade node and removed unused input port. Also, updated ImageWebcam to use LiteGraph.registerNodeType instead of registering it directly.
Added test for non-duplicate sources in CSS imports
Add error handling for Elasticsearch index errors in BulkWriter.prototype.write
Added event listeners for property name and db name changes
Added toggleUi option to Construct and updated apos pushGlobalData to include schemaWidgetsUi with toggleUi value.
Added logging and visibility check for PayPal button rendering
Removed compact mode support in Buffer.rightBrace() method.
Update default settings for exposeExplicitCompositeTask and exposeWithoutPrefix to auto
Removed unnecessary `Buffer.isBuffer()` check in Bloom.prototype.multiCheck
Update ancestor symlink methods to use native fs methods.
Removed unused dependencies: streamify, vinyl-source-stream
Added onSaveError function and updated save functionality to hijack tip label for large bins
Update AutoCompleteList component to use component.props instead of this.props
commit envars fix
Update Laravel Mix configuration for Honey Comb admin panel CSS and add option to disable processing of CSS URLs.
Added workflows for push-renovate-pob_root and push-renovate-prettier, and added a check for existence of .github folder before installing push-renovate-typedoc
Set defaultFocus to false for search input
Added function parseCookies to api.js
Fixes #gh-spin-button: Changed showSpinnerTimeout logic
Fixed issue with device.js where document.body was not properly checked for existence before loading DOM content.
Added didUpdate event handler to sl-grid component for updating heights
Changes to Modal.prototype._handleShown to calculate max height of modal dialog correctly.
 Fixing date constructor to use getFullYear instead of getYear
Added keyboard shortcut for toggleNetworkSelector
Added import statements for _ (lodash) and bindActionCreators (redux) in QueryEditor.js
 Fix for lazy value update when mask is present.
Updated regular expression for splitting path in index.js
Added timer for notification polling
Update adsbygoogle.js to include client parameter for Google AdSense
Update player physics to use THREE.Vector3 instead of game.THREE.Vector3.
Added 'rightHeader' prop to Lane component.
Update search.js to use async/await for YouTube search API call
Added support for using a custom promise implementation in lib.promise(), and added tests for it.
Added support for packet ID 103 and fixed issue with packet ID 240 offset calculation.
Removed unnecessary call to L.DomEvent.fakeStop(e) in L.Canvas.prototype._onClick
Added 'blur' event listener to ContentEditableBinding.
Added timestamp and datetime fields to aofex result object
Added length and md5 properties to readFile and readStream responses.
Increase gecko profiler buffer size from 100MB to 131MB.
Refactoring of updateStructureIndex function and clearStructureIndex function.
Fixed issue with empty function names in instruments.js
Added check for window existence before defining touch events in v-click-outside.js
Fallback to container size when viewport has no size
Added support for ImportDirective and StructDefinition in ContractSource.
Added focus handling for webview browser preloading
Added SHUFFLE_NOREPEAT play mode
Modified internal-plugin-flag to handle missing data in response
Added environment detection to Loki.prototype.copy() method
Update dependencies: Add missing Map polyfill and Object.assign
Added unique ID for each client instance in connect-helpers.js
Fixed bug in stream() function where isString was not being properly reset.
Added support for parsing JavaScript files with ECMAVersion 6 and added colorized error messages for syntax errors.
Move htmlParser assignment from window to elenpi
Fix incorrect promise handling in index.js
Added support for migrations in MongoDB
Added module.exports.readers and module.exports.writers to lib/texts.js
Added check for prerelease version in check-version-md.js
Update authenticationSource.js: Changed text for associated sources
Allow passing a controller name to the render method to use a different controller for the view.
Update minimumInputLength from 2 to 1 in jquery.molgenis.xrefmrefsearch.js
Added 'Column' and 'SlimBanner' to siteIndex array in SiteIndex.js.
Added content security policy and location type to environment config
Fixed bug in gdax.js: Calculate end date correctly for historic rates
Update tcl mode to use chain() for comments and unparsed text
Added function declaration for BlockchainWallet constructor.
Added support for Firefox in Jarallax.js
Update get-artifact-mtimes.js to use process.cwd() instead of resolve(search) for distributionDirectory
Update nightwatchPlugin to use default env and remove output option
Update Platform.js to use npmRegistry.get('/_pluginName/latest') instead of npmRegistry.get('/_pluginName') to get the latest version number
Change name of 'thread' function to 'immediate' and update code to use requestAnimationFrame or timeout for better performance.
Remove unnecessary comment
Added timezone offset to massageColumnValue in scrolltable.js
Update SVGSpriteLoader to use loaderUtils.getLoaderConfig instead of extend for config
Added return statement for vao in GeometrySystem.js
Added global document requirement and updated cache key for Delegator
Added pastebin feature to debugger
Added Coroutine._remove() and _dispose() methods to properly clean up coroutines when they are terminated.
Replace Chance constructor with new Chance() for better seeding
Added ember-flexberry-data version registration
Added check for HTMLInputElement.cloneNode() polyfill
Debounce function now uses requestAnimationFrame instead of setTimeout for better performance.
Update PIDController to use dispKp, dispKi, and dispKd instead of Kp, Ki, and Kd
Added JSON.stringify() to sendWebPush function for sending push notifications with stringified data
Fixed typo in core.js
Updated dojo loader configuration to use async loading and enable combo API. Removed sync loader.
Added protection for vTree in patchNode
Updated extractFilesFromHtml to accept only string values for html and improved error handling.
EnsemblFeatureSource: Remove unnecessary variable declarations and tidy up fetch function.
Added support for dropping columns in SQLite3 tables.
Added pixel ratio and tile size configuration for Plot component.
Update htmlmixed mode to include htmlState in copyState
