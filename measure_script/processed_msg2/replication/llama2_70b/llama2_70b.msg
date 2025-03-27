Catching Throwable instead of Exception in MarshalProcessor and UnmarshalProcessor
Refactored code to validate R identifier in TextEditingTargetRenameHelper .
Added support for executing AsyncTasks in order using Utils.executeInOrder ( ) method .
Added support for decay function with geo fields .
Removed unused import statements in GroovyStructureViewModel.java
Added download content action for PNG and JSON representations in SignavioPluginDefinition .
Added import statement for EnableConfigurationProperties and removed ConditionalOnBean annotation for MongoProperties
Added call to setWindowCallback ( ) on decor ToolbarWidgetWrapper
Added a new case for RANDOM scheduling in RaptorWorkerTimetable .
Added seasons query selection and projection
Added check for &apos; private &apos; and &apos; agricultural &apos; access tags in OpenStreetMapGraphBuilderImpl .
Renamed checkReportsOnApplicationStart ( ) from private to public .
Added volatility to &apos; anyRead &apos; and &apos; anyWritten &apos; fields in Fzip class .
Update ServiceActivatorContext creation to include serviceContainer
Increased size of Dialpad dialog and moved its location
Added import for DefaultModulesProvider in ProjectFromSourcesBuilderImpl .
Added test for round-trip hash code using base encoding .
Added StoreLoad fence in SingleProducerSequencer to ensure visibility of updated sequence values .
Modified ReleasedVersionDistributions.java to filter out milestone versions from the list of supported Gradle versions .
Added updateHideSensitiveForChild method to NotificationStackScrollLayout to hide sensitive content when in ambient mode .
Modified RemoveFailedNodeMessageClass.java to fix incorrect payload byte index in switch statement .
Fixed issue with account initialization and added support for multiple accounts .
Modified AnnotationToRuleConverter to use varargs instead of array
Set text pane &apos;s focusable property to false .
Replace direct buffer with context-specific allocator for Netty responses .
Added import statement for FileDocumentManager and updated actionPerformed method to save all documents before performing HgMergeCommand and HgUpdateCommand
Removed unnecessary map files from BinaryInspector .
Added static final variables for sorting order and updated RealmResults class to use them .
Fixed checkState message to correctly handle null ruleClassName
Removed unused method getPgpData and moved onDecryptDone to MessageCryptoView
Fix InetAddresses.java to use % s instead of hardcoded string &apos; Incrementing &apos; in checkArgument
Added error messages for missing dependencies and scope calculation failures .
Modified DefaultTransform.java to adjust the destination rectangle for proper scaling .
Format mcc value in ResConfigFlags.java to match API level 24 changes .
Added support for escaping Unicode characters in Go string literals .
Added scale parameter to ZWaveThermostatSetpointCommandClass.java and updated payload calculation accordingly .
Added notification settings channels and updated notification setting types .
Added support for rendering return type mismatch errors in override annotations .
Added flag PARSE _ IS _ PRIVILEGED for privileged packages
Migrated restrictions and settings , and updated Toast notification to run on UI thread .
Added test for loading class Echo in ModuleAccessesOSGiServiceTestCase
Modified ToolWindowsWidget.java to update the position of the popup list and set requestFocus to false .
Added test for InodeFile setComplete method .
Fixed bug in SimpleElementValue.java : now correctly returns &apos; true &apos; or &apos; false &apos; for PRIMITIVE _ BOOLEAN constants .
Fix for PyProtectedMemberInspection to correctly handle imports from the same package .
Removed unused menu items and updated preferences handling in GCViewerGui
Added onClose method to HttpClientTransportOverHTTP2 to properly close connections
Added tempFolder creation in setUp method of FileCopierUnitTest .
ZAP : 2016 / 11 / 23 Call postInit ( ) when starting an extension , startLifeCycle ( Extension ) .
Fixed bug in MemberImpl.java where localMember and logger were not being properly initialized when HazelcastInstance was not an instance of HazelcastInstanceImpl .
Added support for opening files with # and file name in EditFileHandler .
Added putInt and getInt methods to ShadowBundle
Update Plot.java to use &apos; set term png small size &apos; instead of &apos; set terminal png size &apos; and fix hardcoded settings .
Fix for incorrect package name calculation when directory is not a directory
Added import for Node from w3c dom package
Add SLF4J support to Redwood logging
Added support for checked exception compatibility constraints in InferenceSession .
Added TransactionSynchronization.STATUS _ ROLLED _ BACK check in SpringTransactionContext.afterCompletion
Update CommentActionResult.java to change COMMENT _ ID _ UNKNOWN from -2 to -2 for replies .
Added support for loading plugins based on their category and module dependencies
Update LocalExecutionPlanner.java to change the value of the hashChannel from 1 _ 000 _ 000 to 1000 .
Updated CommandLineParser.java to include new option --autoDetect and note about variant languages .
Fixed implicit anonymous class parameter highlighting
Added logging for missing out node in switch block
Added test for duplicate functions in different packages and fixed highlighting issue for non-target matching files .
WARNING : Shuffling rows in DRF.java , potentially affecting variable importance
Added constructor for SIPCommTextButton
Update MessageListFragment to properly display subject for threads with no subject
Added @ Deprecated annotation to YUVMonochromeBitmapSource class .
Removed image writing code and added a new sprite for logo .
Added column ORGANIZER _ CAN _ RESPOND to Calendar table .
Modified LafManagerImpl.java to make getDefaultLaf ( ) public .
Added experimental support for data configuration transitions in Skylark rules .
Fixed authentication manager initialization in NetworkCatalogActions .
Update ArrayToString.java to use MATURE maturity level instead of EXPERIMENTAL .
Added test for optimize method in ThreadedEvolutionaryProcess
Added released version 1.0-milestone-8a to ReleasedVersions .
Added scroll indicators to FloatingToolbar
Remove unnecessary import statement in PersistenceContextInterceptor.java.
Added method &apos; newHighlyAvailableFacadeFactory &apos; to class HighlyAvailableGraphDatabase and updated constructor to use it .
GitPushResult : Swapped labels for &apos; beforeUpdateLabel &apos; and &apos; afterUpdateLabel &apos; .
Fixed issue with multiple logins by setting &apos; isPersistent &apos; to false when joining a chat room .
Added jumpToCurrentState and isStateful methods to LayerDrawable .
Added OPTIONS to NanoHTTPD.Method enum
Added check for unversioned files in CommitChangeListDialog
Update default namespace token to use TokenMeta.empty ( ) instead of hardcoded values .
Removed print statement in RandomForest class .
Added support for iterable linked fields in OFetchHelper
Removed call to reportUserActivation ( ) in ABTestEventReportingService .
Remove unnecessary PrivilegeGrantInfo creation and use buildInitialPrivilegeSet instead .
Added import statement for org.hamcrest.Matchers.containsString
EditorUtil : Allow null check for isRealFileEditor and handle null editor in getLastVisualLineColumnNumber
Added TypeReference class for representing generic types
Merge Suggest.java to update scoring logic .
Added validation for non-empty authorities attribute in &lt; provider &gt; tag
Fixed bug in AccessibilityNodeInfo.getParent ( ) : now uses mParentAccessibilityViewId instead of mAccessibilityViewId
Added method getClassNameForJvm to SpringLoadedPositionManager
Remove unused PsiTypeParameterList in GroovyOverrideImplementUtil .
Fixing ScoreboardCommand to correctly handle player names with spaces
Added method setContactUri ( Uri uri , boolean sendToFastrack ) to class ContactHeaderWidget , and updated existing setContactUri ( Uri uri ) method to call the new method with sendToFastrack = true .
Configure internal logging to commons logger .
Added method getModelClass to QueryModelDefinition and ModelViewDefinition to return the class of the model object .
Added journal and raw table master exception messages , updated temp block id messages , and removed unnecessary entries .
Updated GitTaskHandler to prompt user to checkout existing branch when there are problems with the specified branch .
IosHttpURLConnection : updated HTTPMethod assignment
Added setViewVisibility method to FakeRemoteViews
Update Test.java to remove deprecated HTML test report options and use getReports ( ) .getHtml ( ) .setEnabled ( ) instead .
Added support for IPNB format version 2.0 : added &apos; input &apos; and &apos; language &apos; fields to IpnbCellRaw , and introduced CellOutput class for output formatting .
Remove unnecessary AtomicReferenceFieldUpdater for &apos; latest &apos; field .
Added import statements for BeanMetadata and MyConverter
Remove unnecessary icon checks for R Markdown , R HTML , and Sweave modes .
Removed duplicate role addition in UserService .
Fix bug pattern &apos; IsInstanceOfClass &apos; to correctly identify classes instead of class literals .
Publicize ValueWithMetadata.normal ( ) method for alternative MemoizingEvaluator implementations .
Update NotificationPanelView to call updateHeader ( ) after setting stack height in expanding state .
Added catch block for ProcessCanceledException and updated logging in FileBasedIndex .
Added support for onSearchRequested ( ) in WindowCallbackWrapper .
Upgrade Grails core dependencies
Added import statement for grails.util.Metadata and updated logic for determining exported status of plugins .
Added validation for current playback position in TvInputService to ensure it &apos;s greater than or equal to start position reported by onTimeShiftGetStartPosition .
Removed unnecessary method isC0Control ( char ) and simplified isC0Control ( int ) method .
Added EdgeSkipIterator and EdgeSkipIterState to LevelEdgeFilter.java for minor performance improvement .
Replace getInitBody ( ) with getOnCreate ( ) .body ( ) in EActivityHolder # getPreferenceScreenInitializationBlock
Added support for presence in account selection panel .
Fixed issue # 42 : Now using &apos; asNormalisedPublication ( ) &apos; instead of &apos; ivyPublication &apos; for getting configurations .
Update ConfusionMatrix.java to use a different constructor for TwoDimTable .
Added aliases for any ( ) and any ( Class clazz ) methods in Matchers class .
Added functionality to save and update / insert BaseSyncableProviderModel objects based on existence
Update ContextConfiguration annotation in ValidatorTests.java
Renamed &apos; i.heart.ByteBuddy.Object &apos; to &apos; i.love.ByteBuddy.Object &apos; in ByteBuddyTutorialExamplesTest.java
Removed @ Ignore annotation from EmbeddedDatabaseBuilderTests .
Added logging of exceptions in SafeAsyncTask
Added static final fields for default size , precision , and scale , and updated private field values to use these defaults .
Update Channel.java to use ChannelConfig instead of hardcoded configuration parameters .
Fixed bug in ATNConfigSet by removing unused variable and setting configLookup to null when readonly is set .
LoggingExceptionMapper : Changed logException parameter from &apos; id &apos; to &apos; exception &apos; .
Added send count metric for integration channels
Added code to track new inputs discovered by action when executing it .
Added texture filtering for reflection and refraction textures in SimpleWaterProcessor .
Added publish to RPubs option to PlotsToolbar and ViewerPane export menus .
EpisodeDetailsFragment : Remove unnecessary Void &#91; &#93; parameter from FetchArtTask execution
Added a test for expired responses in AsyncProvidersBasicTest .
Implemented consistent exception handling in EventBus by posting SubscriberExceptionEvent when an exception occurs inside a subscriber &apos;s event handling method .
Removed unnecessary logging statement in CookieCutter.java.
Added logger and updated test to handle queue overflow
Update ClusterRule.java to use org.junit.rules.ExternalResource and remove unused imports .
Fixed issue with managed objects being added to RealmList .
Added TextEditorProvider disposal in closeFile method .
Change RMI registry port from 9876 to 37541
Added dependency on ISBN parsing for UPC barcodes
Increased maximum element count for HighPerformanceCache by adjusting the bytes per element factor from 8 to a variable value , and also fixed an issue where the maxElementCount was exceeding Integer.MAX _ VALUE .
Added connect and read timeouts to RealmVersionChecker
Removed unused listener and simplified response handling in AtmosphereFilter .
Added method withDeleteResolver and withMapFunc to PreparedDeleteObject
Added method close ( ) to ElasticsearchResponse to free resources and connection pool .
Update history menu icon from 32x32 to 16x16 .
Implemented OpenStack Swift UnderFileSystem based on JOSS library with directory limitations .
Added center on zero button and reset colors button to ActionsToolbar .
Added check for bundled plugins in InstalledPluginsManagerMain
Modified ComponentProvider and LifecycleListener classes for Jersey 2 Spring integration .
Change weak reference type from OnImageLoadedListener to WeakReference &lt; OnImageLoadedListener &gt; .
Removed unnecessary critical extension OID &apos; 2.5.29.15 &apos; from AsyncSSLSocketWrapper.java and added a null check for cert.getCriticalExtensionOIDs ( ) before removing the OID .
Added waitAtBeginningFactor to RoutingRequest.hashCode ( ) and equals ( ) methods .
Renamed &apos; onViewSelected &apos; to &apos; onTryAndCustomizeSelected &apos; in ThemeBrowserFragment .
Added MIT License header to CanPrintInMultilines.java
Update ElectronBeam animation durations and stretch effects .
Update copyright year in JavaEditorTextProviderImpl.java and add exception for PsiLambdaExpression
Added support for proxy address in ProtocolProviderServiceSipImpl .
Added postCheckIdleStates ( ) call in UsageStatsService to inform listeners of out-of-idle states for all users .
Removed unused case in TestUtils.injectDateInArray ( ) method .
Added methods for creating concurrent soft and weak maps .
Added test for binding with system field error and custom editors
Format string in FMLBukkitProfiler updated from &apos; microseconds &apos; to &apos; % % of overall time &apos; for better readability .
Removed unnecessary code in MeteorChat.java
Added auto-generated flag and setter / getter methods for Axis class .
Renamed print method to print ( Annotation annotation , PrintWriter os , Options options ) and made it static .
Added missing parentheses in TsdbQuery.addPoint method .
Added Cloneable interface implementation to AbstractControl .
Refactor TracIntegrationTest to extend TaskManagerTestCase abstractly and remove unnecessary variables .
Added import statement for CommonRefactoringUtil and updated doGenerate method to check read-only status .
Added import for IntelliJLaf and updated MacIntelliJTextBorder to use Gray.xB1 color for graphite mode .
Added import statement for org.apache.thrift.TException and updated writeSpecifyLocalTest method to throw TException .
Added method getSipUri ( ) to SipPhone class .
Update InternalLoggerFactory to use SLF4J as the default logging framework
Added thread count display for messages with multiple threads
Added &apos; Source on Save &apos; label and updated button texts for small screens .
MCPMerger : Changed usage message and special case for CodecMus
Updated TaskAssigneeTest to use expectProcessEnds ( ) instead of deployer.expectProcessEnds ( processInstance.getId ( ) ) and removed unnecessary line deployer.getTaskService ( ) .complete ( myTask.getId ( ) ) ;
Removed unnecessary Environment creation in Basic3DSceneTest .
Removed unnecessary &apos; conversationVMS &apos; field in MessagesModule .
Added @ Deprecated annotation to SimpleCallback and Callback classes .
Added receiver adapter for cluster peer
Added null check for childContactsNode before accessing its child nodes .
Add static import for Preconditions.checkNotNull and fix verify method signature .
Added broadcast sending for UI update on operation cancellation
Added code to skip adding a person &apos;s email to the list if they are the owner .
Fixed handling of cancel requests in daemon mode
Throw illegal argument exception with correct message
Added breakBlock method to BlockMiningWell to properly break pipes .
Added additional check for areAllBusEventsProcessed in TestResource # areAllNotificationsProcessed
Added error message logging and throwing a ForbiddenException when concurrency level is reached for IndexRangesResource .
Update management center configuration .
Remove redundant trip array and use arrays for stops instead .
Replace &apos; / &apos; with URLS.SEPARATOR in RestIds and URLs
Added new constants and method for determining allowed direction of sliding layer
Fixing issue with list selection after scrolling
Update copyright year in NativeWebRequest.java and switch from request to response objects .
Removed unnecessary logger statement and added missing HttpHeaders construction in HttpProtocol .
Added methods to set useCaches and Accept-Encoding headers in HttpRequest class .
Added new functionality to Context class , including support for multiple constructors and a working directory notion , as well as auto-registration of the block chain object and use of AbstractBlockChain in TransactionConfidence.getDepthInBlocks ( ) method .
Added check for JetReferenceExpression in ExpressionTypingVisitorDispatcher
Added @ Ignore annotation to EJBClientClusterConfigurationTestCase due to intermittent failures .
Added check for canceled status in runWithCheckCanceled method .
Renamed parameter &apos; scheduledExecutorService &apos; to &apos; executorService &apos; in ClusterFactory.create ( ) method .
Added support for identifying problematic classes in MethodDatabase .
Added metrics for vertex index store and edge store operations in InMemoryMetricsTest .
Added @ Nullable annotation to Metadata.get ( Key &lt; T &gt; key ) method .
AddDialogFragment : updated population of show views
Fixed bug in JBossStandaloneEJBContainer.java : deleted temp files correctly
Update IoActions.createFileWriteAction to return Action &lt; ? super Action &lt; ? super Writer &gt; &gt; instead of Action &lt; Action &lt; ? super Writer &gt; &gt;
Added logging for RuntimeShadedJarCreator
Incremented activeTermId and updated activeIndex in Connection.java.
Added method createSoundFromFileDescriptor to SoundFactory , allowing for the creation of sounds from file descriptors .
Added check for fork = = null before running with forked JVM
Removed unused import org.apache.camel.util.Main in MainExample.java
Added method removeAllWithPrefix ( String urlPrefix ) to HttpResponseCache class .
Added support for decompiling Kotlin classes in JetContentBasedFileSubstitutor .
Added path normalization and existence verification to PathTypeCoercer
Added hook for cache invalidation before delete record
Fixed memory mapping error handling in OMMapManager by adding exception logging and reducing max memory on failure .
Removed unnecessary &apos; todo &apos; comments in EditorHyperlinkSupport.java.
Remove unnecessary messages and add timeout for keyguard drawn
Modified GateDeploymentDescriptorTest to compare output and input channel IDs instead of entire channel descriptors .
Added deprecation annotation to RELOAD _ EVENT in Context.java
Added metric fields to DataSchema dimension exclusions
Update CameraManager.java to use &apos; isAvailable &apos; method instead of hardcoded values for device status .
Update JavaQueryTest.java
Added support for hiding Emmet preview hint when editor is released
Removed import of UnsupportedEncodingException in ServletRequest and ServletResponse .
Added check for NaN score in InternalSearchHit.java
Fix for passing correct isLast value to consumer
Added override for performCompletion ( ) and onItemClick ( ) to avoid extra calls from trackball clicks in SearchDialog
Fixed resolve object id for HEAD branch
Fixed issue with persistent presence status not being updated correctly .
Update CLibrary.java to add debug logging for c library functions memmove and bcopy .
Fixed script code constraint attribute in NamedScriptableDefinition
Added setEditing ( false ) to IpnbCodePanel &apos;s updatePanel method .
Remove deprecated method getColumnFamilyStore ( String ) and use getColumnFamilyStore ( int ) instead .
Added Bean Validator component that uses JSR 303 reference implementation ( Hibernate Validator ) and throws BeanValidationException if constraint violations are detected .
Added import statements for Executor and Executors , and replaced ForkJoinPool with newCachedThreadPool in MongoLayerModule .
Added method cancelTaskWindowTransition ( int ) to IWindowManagerImpl .
Added newline character at end of code for R 2.14 compatibility
Removed unnecessary imports in UnifiedParser.java
Removed unnecessary import statement for ZWaveNetworkEvent .
Added method to ModuleVersionSelectionReason interface to indicate if a version was requested in the graph .
Fix for issue # 936 : Removed unnecessary transformation method from search text field .
Remove unnecessary code in SystemServer.java
Added support for serializing view state in WebView.java and ViewStateSerializer.java
Added methods getPsiFile ( ) , getEditor ( ) , and getDocument ( ) to interface HighlightingSession .
Added ability to set performed action for AccessibilityEvent
Added help ID for create new project dialog
FsBlobContainer : Implemented deleteBlob ( ) method using Files.delete ( ) instead of Files.deleteIfExists ( )
Update import statement for AccountUtils from lib.accounts to lib.common.accounts.
Added getUri method to EndpointValidationResult and updated CamelCatalogTest to include additional validation tests .
Updated copyright year in FileReference.java from 2014 to 2016 .
Update ImageProcessor and TexturePacker2 constructors to accept a null rootDir parameter .
,1
Replace special format char &apos; % &apos; with &apos; % % &apos; in log messages
Added warning that AutoValueExtension API is not final and will change in a future release .
ConfigTest.java : Updated assertion for iOS Info.plist file
Added import statement for PluginManagerCore and updated version check for Kotlin plugin
Added null check for serviceUuid in BeaconTransmitter
Clear usages cache in UsageViewImpl
Allocator interface updated to use String view instead of int view
Added CDDL header and updated license information .
Added logging statements to DnsUtilActivator
Update DesktopIdentification.java to set &apos; neo4j.ext.udc.source &apos; property instead of &apos; neo4j.ext.udc.launcher &apos; .
Fixed initialization of Box class by checking for invalid coordinates .
Added &apos; removeGuestPI &apos; and &apos; logoutPI &apos; intent to user switcher notifications
Added check for fullscreen mode in sanitizeOverrideConfiguration method .
Added writeable callback for AsyncSSLSocketWrapper
Added import statement for ApplicationInfo and modified code to use PackageManager.GET _ UNINSTALLED _ PACKAGES to get resources for application icons .
Added null check for image URL in getImage method of ResourceManagementServiceImpl .
Remove unused configuration options TITAN _ HADOOP _ GRAPH _ OUTPUT _ TITAN and TITAN _ HADOOP _ GRAPH _ OUTPUT _ TITAN _ INFER _ SCHEMA
Modified Method.java and JSONModel.java files .
Removed unused imports : org.jboss.as.server.services.net.ManagedBinding , org.jboss.as.server.services.net.SocketBinding , and org.jboss.as.server.services.net.SocketBindingManager
Added ErrorProneScanner with DEFAULT _ CHECKS and updated the scanner injection mechanism .
Fixed issue with compression detection in CompressionTest .
Fix for POSTGRES array handling
Added test for unregistering extension point in EntryPointsConverterTest .
Fixed syntax highlighting for GrVariableImpl.java
Added import statement for BaseElementAtCaretIntentionAction and updated CreateSwitchIntention class to extend BaseElementAtCaretIntentionAction instead of PsiElementBaseIntentionAction .
Updated kill query command format for ExasolServerSessionManager .
Increased maximum doze duration from 10 seconds to 60 seconds .
Added import statement for Arrays class and updated copyOf method to use Arrays.copyOf instead of System.arraycopy
Trim leading and trailing white space from cell values before importing .
ImageView : updated setImageResource ( ) method to always load the resource even if the resId hasn &apos;t changed .
Added method getServiceConfig ( String ) to class Services .
HexDumpProxyFrontendHandler : Remove unnecessary suspension of incoming traffic .
Added missing brace in &apos; testDontRemoveBreakInTryFinallySwitch &apos; test case .
Fixed cyclical references in UniversalEnglishGrammaticalStructure .
Added error handling for BluetoothService dump method
Added menu item for changing nickname in chat room
Added check for upper ell in long constants ( Java Language Specification , Section 3.10.1 ) . Changed comment to include rationale and example configuration .
VcsLogDateFilter : changed date comparison from authorTime to time
Added exception message with image dimensions in DefaultImageRaster.rangeCheck method .
Remove entry from session cache
Added line breaks for better readability in OConsoleDatabaseApp.java.
Fixed issue with wildcard handling in AntPathMatcher ( SPR-13139 )
Added apk add --update bind-tools command in HeliosSoloIT.java
Update import statements for org.gradle.util and org.hamcrest packages
Removed unnecessary import of XposedBridge and removed log statements that were not needed .
Removed unused drag icon variable .
Added test for empty header lines in HttpMessageDecoder
Added temporal annotation to JpaTraceEventMessage class
Added JavadocStyleCheck to ensure that documentation comments have a period at the end of the sentence .
Initialized new component and routed it to the appropriate service .
Added check for null before setting carrier in getDataUsageInfo ( ) method .
Replace method reference with context-dependent candidate
Added error handling for unexpected errors during index update
Added Ignore annotation to JMXMonitorTypeLongCounterTest due to missing monitorNotificationLong.xml file .
Added support for annotating plain text files in AnnotateRevisionAction .
Added ability to unpack fat jars at runtime
Disallow shared user change during package installation
Added stopAllSounds method to AudioManager and stopped all sounds in World.stop method .
Added method getArg ( int index ) to class FeaturesTest and modified getPattern ( ) method to use myCustom ( ) function .
Added seal method for inverted index creators .
Added test for index hits from querying removed nodes , and fixed issue with negative count in query results .
Modified PropertyEventBus.java to remove unnecessary onInitialize call for subsequent subscribers .
Removed import of CheckstyleException in DebugChecker.java
Removed unnecessary recycling of mapping data in CoyoteAdapter .
Updated Android Maven Plugin to use &apos; Android Maven Plugin ( JVM ) &apos; instead of &apos; Maven Android Plugin ( JVM ) &apos;
Fixed bug in GitQuickListContentProvider by adding a null check for action
Fix for issue # 3003 : Auto-format undo not working properly
Update Groovy Compiler Customization Builder documentation link
Added default ctor check for json parsing
Added downcasting for derived exceptions in TachyonException
Added transitory tags to TaskListFragment &apos;s empty state
Added scrollValuesObsolete flag to LightGrid.java
Added option to disable DOT file generation in RunCamelMojo
Update event bus test to use MyEvent class instead of String class
Added File conversion for platform-specific resource path in TestVirtualContext .
Update LanguageAction.java to support additional operating systems .
Update ManagementCenterService.java to use a new URL for the management center .
Removed unused method getKeepAliveLeft from SocketWrapper
Added check for channel registration before setting executor
Fixed issue with expire store size in CacheOffHeap class .
Fixed incorrect return value in VanillaInventoryCodeHooks.java
Added field references for OnMethodEnter skip and prepend line number
Change log level from debug to error for better error handling in CommandContext .
Remove NonDefault serializer feature
Added AGPL license header to MathJaxLoader.java
Fixed logging message for non-existent external .resS file in AudioClipHandler .
Changed type of &apos; destKeys &apos; field from &apos; Key &#91; &#93; &apos; to &apos; Key &lt; Frame &gt; &#91; &#93; &apos; in class SplitFrame .
Rename class from LocalInspectionTool to BaseJavaLocalInspectionTool and add missing annotations .
Added support for switch component with text size adjustment .
Add return statement for MethodInvoker.invoke ( ) method .
Added IllegalArgumentException and IllegalStateException checks for unsupported crypto operations in FingerprintManager.authenticate ( ) method .
Added check for mAction ! = null before exposing file URIs in Intent.java
Added garbage collection of objects with locks on files in Library class .
Fixed issue with autoStartsEdit mode not working properly with non-typed events .
Remove hack for class file v53 and update test exclusions in JavaPluginConvention .
Increased throttling time from 200ms to 750ms in ThrottlerTest .
Added support for WebSettings.LayoutAlgorithm in TestWebSettings .
Added override of toString ( ) method in Crouton class .
Added error code for contact already exists and updated MetaContactListServiceImpl to throw MetaContactListException when contact already exists
Added support for formatting item labels with % d specifier .
Added @ Nullable annotation to Function.apply ( ) method .
Fix copy resource ID method to return void and remove unnecessary return statement .
Added support for checking unused enum and annotation definitions in UnusedImportsCheck .
Added ensureYellow ( ) call in TransportSQLActionTest.java
Added import statement for IndexMetaData and updated createIndex method to use prepareCreate with settings .
Fixed issue with deferring scans on Lollipop devices by using UsageStatsManager to check if app is inactive
Added tests for EL escape sequences and string literals .
Added check for sequenceFlows.containsKey ( sequenceFlowId ) before getting sequenceFlow from sequenceFlows map .
Update BloomFilter.java to use ColorRGBA.BlackNoAlpha instead of ColorRGBA.Black for background color in glow mode .
Added import statement for TinkerPop &apos;s Gremlin process and modified vertex creation to use TinkerPop &apos;s API .
Added logging for unknown drawer states in MenuDrawer .
Add logging of DNS event types , return codes , and latencies for each network interface
Minor change in BaseIndex.java : Changed log level from info to debug for INSERT actions .
Fixed issue with AutoSet params not being properly set in Param class .
Added reinit method to PresencePanel and called it in SIPAccountRegistrationForm
Added setCustomLoadMoreView and swipeCustomLoadMoreView methods to UltimateViewAdapter .
Added method to dump alignment information in formatting model .
Added test method &apos; testIt &apos; to class TestJsonPointer .
Added method setMaxHeight ( int ) to set maximum height of scrollable menu
Added check for presence status before registering protocol provider .
Fixed bug in database cache : updated file history ID for new file version
Modified XmlPluginDescriptorReader to use plugin.xml instead of .
Fix for CAMEL-12345 : Do not delete file if process fails
Fixed scaling issue for DeferredIconImpl by checking if the delegate icon is scalable before updating the scale factor .
Added gc ( ) method to System class .
Added ability to create and manage run configurations for items in the execution menu .
Removed constructor and set numRecordsPerBlock _ to a constant value
Added code to handle existing block store files and improved error handling in BoundedOverheadBlockStore constructor .
Added countdown latches for Tachyon master container allocation and application completion .
Added check for invalid FoldRegions in LivePreview
Modified XMLSerializer to use &apos; dc : identifier &apos; instead of &apos; id &apos; for book URIs , and added error handling for SAXExceptions in book and bookmark parsing .
Modified SettingsMenu.java and StatusSubMenu.java to change the behavior of the &apos; Enter &apos; key in chat windows . Now , pressing &apos; Ctrl + Enter &apos; will send a message instead of just &apos; Enter &apos; .
MavenProjectImporter : Fix for disposing modules
Added logic to handle filling of RecyclerView when scrolling in both directions .
Removed unnecessary import statement and removed @ NotNull annotation from adapter constructor parameters .
Added exception handling for ClientAbortException and IOException in StandardHostValve.java.
Added script source release in TemplateCompiler .
Replace ServletUpgradeRequest and ServletUpgradeResponse with ServletWebSocketRequest and ServletWebSocketResponse respectively to use Jetty 9.1
Update ClassLoaderLogManager.java to use correct class name in DEBUG _ PROPERTY constant
Modified CassandraUnitUtils.java to change the port of embedded Cassandra from 9042 to 30000 .
Added missing unique constraint and index changes to DiffResult .
Added variable &apos; distance &apos; to SavingTrackHelper and set it to 0 .
Remove unused dependency descriptor mediator for version mediation of transitive dependencies
Added check for empty active notifications map before showing individual notification for builder .
Added field &apos; myAllFields &apos; to &apos; CodeStyleSettingsCodeFragmentFilter &apos; and modified &apos; FilterFieldsTask &apos; constructor to include it .
Removed unused import : java.util.List
ClientEncodingFilter : Changed return value of applyEncoding from true to false
Modified Task.java to remove unnecessary modifier
Removed unused import statement for &apos; FileTrackingOperationDto &apos; and added check for workspace being stopped before saving a file .
Fixed issue with upload client socket . Now creating a new client object for next uploads when an exception occurs .
Added support for right margin settings in Groovy and Python code style providers .
Fixed issue with duplicate work-only apps showing up as &apos; not installed &apos; for some users by removing owner entry if it isn &apos;t installed .
Added logic to strip type prefixes for invalid code removal
Added log statement for received raw message from CUL and removed unnecessary EOB check .
Removed unnecessary &apos; intercept &apos; method implementation in ClassImposterizerTest and VerifyingRecorderTest .
Terminating connection after failure to send message to client . This may be because the client has gone away ( see https : / / java.net / jira / browse / SERVLET _ SPEC-44 ) .
Added import statement for com.sun.jdi.request.EventRequest
Renamed ComponentModelBasePlugin.apply ( ) to ComponentModelBasePlugin.getSimpleName ( ) .apply ( )
Renamed toString ( ) method in FilteredTraverserBase to include class name and traversal information .
Update HpackEncoder.java , dynamicTable is now a HashMap &lt; HttpString , List &lt; TableEntry &gt; &gt; instead of a Map &lt; HttpString , List &lt; TableEntry &gt; &gt;
Fixed spelling mistake in HttpClient.java
Fixed calculation of line width in BitmapFont.java
Fixed EnterAction visibility and enabled state for non-image files .
Added download start listener
Added StrictMode.disableDeathOnFileUriExposure ( ) in ResolverActivity to prevent crashes when handling intents from legacy apps .
Added method collapse ( ) to collapse all expanded items and added collapse ( int position ) to close a specific collapsible item .
Added support for concurrent injection by replacing LinkedList with LinkedBlockingDeque in InjectableObjectFactory .
Removed unused import statement for SQLiteStatement in LocalStore.java and Storage.java
Added sensitive logging for IMAP commands
Added support for TripleDes encryption in Nagios component .
Changed timeout value in ApplicationImpl from 30 minutes to 5 minutes .
Renamed SerializedConverterTest to QueueFileLoadingTest and moved it from src / main / java to src / test / java , and also renamed SerializedConverter.java from src / main / java to src / test / java
Added error handling for asset upload errors and recording failures in ConversationFragment .
Added initialization method for DBFlow and removed unnecessary release of resources in FlowManager .
Added MediaType.KEY _ ARCHIVE constant for PKCS # 12 key archive format .
Added setting listener to Application class
Added ability to sync calendar and contacts for call log backup .
EditShareFragment : Handle share not found error and refresh UI
Added &apos; isRejoining &apos; method to CollabEditStartParams class
Added abstraction to FilterStreamInput by making it an abstract class .
Fixed typo in MessageProtocol.java : changed &apos; clas &apos; to &apos; class &apos; in exception message
CacheManagingDrawTask : updated logic for repositioning caches
Added test for stubbing with throwables
Added shard Ids to PlainOperationRouting for better estimation of operation execution .
Added check for physical file existence in PsiClassImplUtil
Changed intent type from ACTION _ PICK to ACTION _ GET _ CONTENT in prepareVideoLibraryIntent method .
Update NavUtil.java to use Linkify.WEB _ URLS &#124; Linkify.EMAIL _ ADDRESSES instead of Linkify.ALL
Added &apos; closed &apos; field to SyncingFileSystem and added check for closed status before syncing and closing the file .
Added import for Providers class in ClusterDynamicSettingsModule
Changed CLIConfigurationKeys.MESSAGE _ COLLECTOR _ KEY from MessageCollectorPlainTextToStream.PLAIN _ TEXT _ TO _ SYSTEM _ ERR to MessageCollectorPlainTextToStream.NON _ VERBOSE
Added includeId parameter to CSVResultsExtractor and updated extractResults method signature .
Remove unused variable and shutdown executor service in FileSystemMasterIntegrationTest .
Update UsersController to use user.getPermissions ( ) instead of formData.permissions
Fix bug in PullToRefreshBase : weight vs width
Animation.java : Changed onAnimationEnd listener to be called only once per animation
Fixed issue with incorrect modification stamp in SingleRootFileViewProvider .
Removed unused import and simplified Google Analytics tracking in BaseTopActivity .
Added support for handling multiple contention actions for a given lock .
Added dependency on spring-integration-dsl-groovy-core 1.0.0.M1 and updated SampleIntegrationTests to use it .
Fixed issue with PanelView &apos;s touch slop exceeding initial offset .
Removed unnecessary code and optimized bubble sorting in MainController .
Added check for module existence in PluginRunConfiguration
Added methods onRemoveBlockByClient and onRemoveBlockByWorker to LRUEvictor class .
Added try-catch block for FileNotFoundException in ModelSpecificDistanceCalculator.java
Added deleteOnExit method to File class and registered it with atexit ( ) for proper cleanup on exit .
Replace polygon triangulation algorithm with a better one using glu tesellator code from PGraphicsOpenGL
Remove unnecessary getLogger method
Fixed flag in BarController.java to use SYSTEM _ UI _ TRANSPARENT instead of SYSTEM _ UI _ FLAG _ FULLSCREEN
Added import statement for com.google.common.base.Throwables and updated recoveryFailed method to use Throwables.getRootCause ( ) to get the root cause of the exception .
Update ContributionManagerDialog to display a progress bar while downloading the contribution listing .
Fixed test method name from &apos; testEquansIgnoreCase &apos; to &apos; testEqualsIgnoreCase &apos; in HttpHeadersTest.java
Added debug names to MemberType classes and updated toString ( ) method to return the debug name .
Added OsmandExtraSettings and AccessibilityPlugin , removed OsmEditingPlugin
Added import statement for ActionMethodParamNameResolver and registered it as a component .
Added method clearCache ( ) to GrailsViewResolver and overridden superclass method to clear view cache .
Replace HEAD with ticket.mergeTo in TicketPage.java
Removed unnecessary imports : GWT , ClientBundle , ImageResource , and Provider .
Added WebListener annotation to ModelManagerListener and removed unnecessary application listener registration
Removed unnecessary initialization of stores in ReadOnlyStoreManagementServlet .
Removed unused method &apos; itemsFormat ( JsonFormatVisitorAware toVisit ) &apos; from ArraySchemaFactory and JsonArrayFormatVisitor .
Renamed class &apos; Attribute &apos; to &apos; InternalAttribute &apos; in file &apos; io / netty / handler / codec / http / multipart / InternalAttribute.java &apos; .
Added user permissions validation and auto-generated password feature .
Update SinaWeibo2Example.java to use dynamic api key and secret
Added import statement for android.text.TextUtils and updated onProgressChanged method to check for non-empty title before updating activity title .
Added test for outer join with common expression
Added flag MOVE _ TASK _ NO _ USER _ ACTION to moveTaskToFront method
Renamed method names in ISO8601DateFormatTest to make them more descriptive and consistent with Java convention .
Added Apache License header to CommandUtils.java.
Added VisibleForTesting annotation to clearLoadedModules and getLoadersMap methods .
Updated HttpClientTimeoutTest to use badport instead of connector.getLocalPort ( )
Added method featureValueOf ( T actual ) to class FeatureMatcher &lt; T , U &gt; to extract the interesting feature from the target object .
Remove type parameter D from method collectManagedProperties ( StructBindingExtractionContext , Map ) in DefaultStructBindingsStore
Added import statement for GitUtil and refreshFiles method call in GitPushActiveBranchesDialog .
Update RoutePreferencesMenu to use routingHelper.getAppMode ( ) instead of settings.APPLICATION _ MODE.get ( )
Replace Roxygen example range regular expressions with more generic ones .
Fixed calculation of down pre-scroll range in AppBarLayout .
Fixed StreamBody.write method to pass correct content length to Util.pump
commit message here
Added check for disposal before running tasks in AbstractTreeUi
Fix for updating widget positions in dashboard
Added import statements for SeriesGuidePreferences and R class , and updated ListWidgetConfigure to use notifyAppWidgetViewDataChanged method .
Removed unnecessary @ GuardedBy annotation from SubscriberRegistry class .
Updated TcpFailureDetector to use longValue ( ) when calculating timeIdle
Added confinedToStage and forward parameters to ManagementGraphIterator and updated logic for iterating over management graph .
Removed unnecessary ExecutorService finalizer and replaced it with AsyncWriteService for resuming connections .
Removed display of points in GPXLayer
Added getPreferredSize ( ) method to BranchesPanel to improve performance .
Added newline character to Log.warn statement in SocketConnection.java for better readability .
Added import statement for Logger and added logging statements in GitRevisionNumber class .
Skip JDK classes when checking for superclass
Allow null path info in HTTP requests .
Increased ID authority wait time from 10ms to 100ms .
Added getRootFile method for getting canonical file path and updated storeDir and migrationDir with getRootFile method .
Removed IterationListener interface implementation from OutputLayer
Add SonarQube copyright and license header to IndexSynchronizer.java
Fix for moving stack that was not added
Removed unnecessary &apos; writer.flush ( ) &apos; call in FileHandler.java
Added PoolSizeViolationException with improved error message
commit - Fix for infinite loop in NioEndpoint.java
Fix for invalid drop-result claims during drag and drop operations .
Added timezone information to SystemResource and ServerStatus
Added empty map for services in FileSystemWorker
Replace drawPath with drawCircle for outer circle .
Added test case for verifying no more interactions in order
Renamed &apos; setType &apos; to &apos; setDataType &apos; in OracleTableColumn class and updated the corresponding method signature .
Added closing of data stream and file system in BSONSplitter .
Updated MainController.java to set pivot for content view based on selected bubble .
Throw IllegalStateException instead of RuntimeException when loading class fails .
Added capacity check and corrected list initialization in MethodLibrary .
Added method getWorld ( ) to interface Chunk .
Added global status selector box for non-hidden providers
Removed unnecessary null check in removeChangeListener method .
Added tests for BitmapFactory.decodeResource ( ) and decodeFile ( ) methods .
Change ISODateTimeFormat.getInstance ( ) to ISODateTimeFormat.getInstance ( chrono ) in StringConverter.java
Added logging statement in Webapp.java for development mode detection
Added filter execution summary with name , status , and time for RequestContext
Added exception message with POJO name in OObjectSerializerHelper
Ensure native finalizer is not invoked if constructor threw exception
Update ViewPagerActions.java to change the description of moveOnePageToTheRight to moveToASpecificPage
Added test case for sorting list of strings with custom comparator .
Added editRmdFormatOptions command to TextFileType and made it available when RMarkdown package is installed and active editor is an RMarkdown file .
Added test case for JSONPObject : test _ jsonp _ 1
Fixed bug in ExopackageDexLoader : now correctly deleting stale odex files .
Added method dispatchSetPressed to prevent ViewGroup from dispatching pressed state to its children .
Modified SignavioFileSystemConnector to correctly handle artifact IDs with .signavio.xml extension .
Fixed crashes occurring with certain languages ( Spanish , Polish ) and backup service deleting too many old days backups .
Renamed constructor parameter from &apos; verticalDivider &apos; to &apos; divider &apos; in class DividerItemDecoration .
Removed unnecessary imports : JCheckBox , JComponent
Update ReactTextShadowNode to use ceil instead of round for calculating image size
Removed unnecessary import statement for &apos; AsynchronousService &apos; in ChannelService.java.
Bumped version number from 0.9.9 to 1.0.0 .
Added check for root id in InodeTree # isRootId
Added check for null config in LruBitmapPool.put ( ) to prevent using bitmaps with null configs that can &apos;t be reused .
Added import statements for FML relauncher and modified handleBlockHighlight method to include @ SideOnly ( Side.CLIENT ) annotation .
Disable backend ping for Android versions below ICS ( Ice Cream Sandwich )
Added check for redeclaration diagnostics and marked declarations with red highlighting .
Added permission check for scheduling jobs that will be foreground
Modified Peer.java to use &apos; addr &apos; instead of &apos; address &apos; for consistency .
Added clear method to FragmentSeparatorsPositionConsumer and called it in updateEditors of FragmentedDiffPanelState .
Fixed issue with incorrect display of numeric values in ResultSetViewer
Added support for displaying a countdown timer when undoing an item removal
Added import statements for CleanupHelper and assertions in CliTest.java.
Fixed issue with empty module list
Fix for camera device close issue
Fix for ViewGroupOverlayUtils &apos; moveViewInOverlay method to handle non-zero screen coordinates .
Removed unnecessary import statement for &apos; containsLine &apos; in BuildScriptErrorIntegrationTest.java
Added tests for URI decoding bugs ( # 54602d , # 54602e )
Added check for OpenGL ES support
Fixed issue with ShowUsagesAction enabled state
commit message goes here
IgnitedAsyncTask : removed unnecessary varargs parameter and added typesafe execute ( ParameterT arg ) method
Increased heartbeat frequency from 5 seconds to 30 seconds in HeartbeatInterceptor .
Deprecated ConfigAttributeEditor class .
Removed unnecessary startActivity call in MainActivity .
Added import statement for ToolWindow and removed unnecessary import statement for ToolWindowImpl .
Added import for JavaResolveUtil and updated FunctionalInterfaceSuggester to use it .
Fix wakeup requirement level checking in NioProviderMetadata
Added test source content filtering to AnalysisScope
Added check for showing chat rooms and recent messages in PresenceFilter
Changed host from &apos; 127.0.0.1 &apos; to &apos; localhost &apos; in AxisReportIncidentServiceTest .
Replace &apos; assertTrue ( latch.await ( ) ) &apos; with &apos; assertOpenEventually ( ) &apos; in BasicMapTest .
Changed status color from black to UIUtil.getLabelForeground ( ) for unrecognized files .
Added @ since 2.5 annotation to SentenceWhitespaceRule.java
Show progress bar while submitting comment
Update log message in Database.createDatabase method
Modified DrawHandler.java to change quit ( ) method implementation .
Added test for MapReduceOutput.getOutputCollection ( ) and fixed issue with dropOutputCollection ( ) not dropping collection properly .
StatsActivity.java : Removed unnecessary constructor in StatsWidgetPromoDialogFragment .
Fixed issue with VirtualFileWindowImpl &apos;s isWritable ( ) method and added missing setWritable ( ) call in MultiHostRegistrarImpl .
Changed visibility of introduce constant popup from true to false
Added getPreferredSize ( ) method to JBLoadingPanel
Fixed error message in SQLCreateIndexTest to include correct index name .
Added support for nested resource handling in ProjectRegistry .
Fix for PythonTask.java : Remove unnecessary check for Windows and remote Python interpreters
Remove unused variable &apos; expectedResolveData &apos; and move its initialization to the method &apos; getExpectedResolveData ( ) &apos; .
Added support for setting time request JID in EntityTimeManager
Fixed validation for URL field in ContactInfoGenerator .
Fix for SipPhone # equals ( ) method to correctly compare SIP URIs .
Initialized numAllowedChannels in WifiStateTracker after supplicant is alive
Added &apos; final &apos; modifier to CalendarSerializer and DateSerializer instances .
Increased MAXFILESIZE from 64000 to 64000
Update configuration class to use termBufferLength instead of termCapacity for publication window length .
Added test for large requests in BatchOperationFunctionalTest
Added import statement for org.junit.Assert and added getOnlyConstructor method to TestClass .
Added empty layout params to prevent crash in TextView
Added import statement for com.intellij.lang.ASTNode
Modified OServerCommandGetStaticContent.java to return true instead of calling super.beforeExecute ( iRequest , iResponse )
Removed unnecessary stop from transit route when showing intermediate stops and alighting .
Added import statement for com.intellij.ui.treeStructure.Tree
Added import statements for IsoChroneSPTRendererAccSampling and IsoChroneSPTRendererRecursiveGrid , and bound instances of these classes in OTPConfigurator .
Update GLRenderer to set viewport resolution to match framebuffer
Added test for double-decoding attribute values .
Added getConfusionMatrix method to Evaluation class .
Added check for containing whitespaces only in FormatterUtil .
Ensure that QueryImpl can handle non-iterable types
Added flag PendingIntent.FLAG _ UPDATE _ CURRENT to setContentIntent ( ) in VolumeUI.java.
Incremented numObjects by 1 instead of count for more accurate statistics
Renamed &apos; toLocation &apos; method parameter from &apos; teleportLocation &apos; to &apos; location &apos; .
Added idempotent reset mechanism for HystrixTimer
Modified PrintActivity.java to update print job info
Fixed bug in ReaderAttachmentTable by removing unnecessary transaction
Added tearDown method to TestPersistentInterval class
Removed unused import org.objectweb.asm.commons.RemappingMethodAdapter
Do not load content for unknown file types .
Added support for retrieving annotation attributes from meta-annotations .
Added NullOutputStream.java , which throws away all output instead of outputting anything , as suggested in http : / / stackoverflow.com / questions / 2127979 . Author : John Bauer
Fixed bug in RuleIndex.java where empty query strings were not properly handled .
Removed unnecessary import statement for tachyon.master.IndexedSet
Added logic to convert 99 % to 100 % for Z-Wave multi-level switches
Added check for partitioning or virtual column in SemanticAnalyzer .
Added ability to run CMake commands and print library folder content .
Updated Patterns.java and PatternsTest.java to add additional validation for phone numbers .
Added closing parenthesis in SelectQueryRunnerTest.java
Added attribute service initialization for root directory .
Fixed error in ScriptIntrinsic constructor
Added override of toString ( ) method in FirebaseMessage class .
Fixed error in FacebookSdk.java where app Ids were not properly formatted .
Fix RouteRecalculationThread to stop calculation when parameters are not changed
Added method clearActor ( ) to Cell class .
Added return statement at the end of SurroundWithHandler.doSurround method .
Added SecurityCheck annotation to UserREST.java
Fix for concurrent unsubscription issue in OperatorPublish
Added method getAll ( ) to Directory interface and implemented it in GHDirectory class .
Added import statement for ISODateTimeFormat and updated timestamp parsing method .
Update TestNGUtil.java to use a random UUID for the cloud name and fix the ice root path .
Renamed intent extra field from ACTION _ DOCK _ STATE to EXTRA _ DOCK _ STATE .
Added check for valid user before clearing calling identity in InputMethodManagerService
Added &apos; final &apos; keyword to variable &apos; instance &apos; in method validateCheckSection
WebView : updated scrollTo ( ) method to take into account visible title height when scrolling .
Added SpringCamelContext integration and enabled consumer template caching .
Added trim ( ) method to remove excess semicolons in DatabaseHelper.java
Added static import for REPLACE _ EXISTING and updated Files.copy method to use it .
Added exception message for missing deserializer in readValue and readPropertyValue methods
Remove unnecessary @ GwtCompatible ( serializable = true ) annotation in ArrayTable .
Added settings for performance logging and legal terms of service
Renamed Characoters.java to Characters.java and updated swipeToLoadLayout.post in ClassicStyleFragment .
Remove unnecessary check for empty columns in NativeMetadata
Fix for Java bug with unchecked warning in StoreDefinitionsMapper.java
Added import of reactor.Publishers and changed return type of handleResult method to Publishers.completable
Remove unused JdbcSqlStat entries in JdbcDataSourceStat .
Increased shutdown timeout from 3 seconds to 30 seconds in BlockingHttpEndpointTest .
Renamed AccountHelper.java from util package to models package and updated import statement and variable names accordingly .
Added import for JetFileType in CodegenTestsOnAndroidGenerator.java
Added test for instance of check with GETELEM and fixed bug with prototype chain
Added support for formatting Javadoc comments with arguments .
Fixed issue with soft wrap advance when document changes .
Merge ( destination , constituents ) : changed &apos; toArray ( new FileBackedCache &#91; constituents.size ( ) &#93; ) &apos; to &apos; toArray ( ( FileBackedCache &lt; KEY , T &gt; &#91; &#93; ) new FileBackedCache &#91; constituents.size ( ) &#93; ) &apos; .
Added check for if-then-else without curlies in RightCurlyCheck .
Added getHttpContext ( ) and setCookieStore ( PersistentCookieStore ) methods to AsyncHttpClient
Added editor.onCursorChange ( ) to fix console scroll pos jumping on first typing .
Removed @ Test annotation from testAutoCommit method in ConnectionStateTest.java
Added method standaloneSetup ( DefaultMockMvcBuilder ) to RestAssuredMockMvc class
Added copyright header to EnvironmentObjects.java.
Added import statements for &apos; Model &apos; and updated interfaces to extend &apos; Model &apos; .
Added @ SuppressWarnings annotation to avoid unchecked cast warning in DefaultAccessTokenConverter.java.
Fixed synchronized block for serial port write
Added test for sending 0 length data packet after flow control timeout .
Update FolderList.java to change scrollbar style from OUTSIDE _ INSET to INSIDE _ OVERLAY .
Added support for introducing constant expressions in DuplicateStringLiteralInspection .
Added IntegrationTest property to differentiate regular tests and @ IntegrationTest tests
Updated Roster.java to use a more efficient way of waiting for the roster to load .
Added note to MongoCollection and MongoDatabase interfaces regarding binary compatibility .
Added support for debugging with JPDA on remote machine .
InspectionEngine : added public accessibility for Upsource and updated inspectEx method signature
Update NettyAsyncHttpProvider to use MAX _ BUFFERRED _ BYTES for dynamic buffer allocation .
Fixed issue with drawing images at correct scale .
Fix leak by using sentinel in ReferenceQueue
RestoreOperation : Implemented file history and versioning for restore operation
Added support for function aggregates in SELECT statements .
Added @ SuppressUnderAndroid annotation to MacHashFunctionTest.testNoProviders ( ) to suppress a warning about sun.security.provider being unsupported on Android .
Fixed an issue with HystrixThreadPool where the core pool size and maximum pool size were not being set correctly when using fast-properties . Also added a check to avoid setting the pool size if it hasn &apos;t changed to prevent unnecessary lock operations in JDK 6 .
Fixed issue with ManualRangeMarker.applyEvents ( ) not being called when it should be .
Removed unnecessary createToolTip ( ) method and fixed tooltip display for ChatConversationPanel .
Fixed issue with Windows process management by correctly retrieving process ID .
Removed unnecessary imports and removed request for window features .
Modified WindowState.java to change the fitToDisplay logic .
Added new fields to GwtApplicationConfiguration : width , height , stencil , antialiasing , fps , rootPanel , canavsId , log , useDebugGL
Removed unused import statement for &apos; org.joda.time.DateTime &apos; in SingleSignOnSessionsReportController.java.
Fixing IN operator error message for indexed columns .
Disable PyClassicStyleClassInspection by default
Added synthetic links enabled setting and synchronized setter method .
Renamed TABLE _ MODEL _ NAME to TABLE _ MODEL _ NAME in TableModelFileWriter.java
Removed unnecessary code in WallpaperService.java
Update App URI scheme from &apos; android-app &apos; to &apos; android-app _ &apos; in MainActivity.java
Removed unused import statement for &apos; Collections &apos; in EipAnnotationProcessor.java.
Fix for corrupted class files : updated ClsFileImpl to use a SoftReference for stub Tree
Added support for creating classes with a superclass in the OServerCommandPostStudio .
Removed unnecessary &apos; equals &apos; and &apos; hashCode &apos; methods from CommitLogSegment class .
Removed unnecessary disconnect ( ) call in WelcomeActivity &apos;s onDestroy ( ) method .
Disable XmppConnection debug logging
Fixed exception handling in PSurfaceNEWT.java
Update ExpectedHighlightingData.java to change regular expression for highlighting types .
Remove unnecessary enableDefaultOffRule call
Updated DataInputStream.java to correctly implement readUTF ( ) method .
Added @ Ignore annotation to PerformanceAndSanityIT.makeSureFilesAreClosedProperly ( ) to avoid flaky test .
Dispose of old root model when replacing it with a new one .
Added @ SuppressWarnings annotation to makeAnnotation method to suppress unchecked warning .
Added display of currently active wake-up interval in sleep mode stop dialog
Added call to ApplicationManager.getApplication ( ) .invokeLater to ensure PsiDocumentManager.getInstance ( project ) .commitAllDocuments ( ) is called in the UI thread .
Added support for voice messages in JsMessage entity .
Added epsilon term to MultiLayerNetwork , and added methods for accessing and modifying it .
Added check for zero max walk distance in RoutingRequest.setMaxWalkDistance ( ) method
Added security check for missing session and token in ONetworkProtocolBinary.java
Added support for displaying YAML titles in DocumentOutlineWidget .
Added parantheses around selection criteria for SQL validation
Added handling of single account selection and empty adapter case in AccountSelectorActivity .
Modified GdxSetupUI.java to use system-dependent SDK location .
Modified ButtonDispatcher.java to return View.VISIBLE instead of mVisibility and to return 1 instead of mAlpha when mVisibility or mAlpha is null .
Added input type and validation for email preference
FavoritesManager.java : Changed type of children collection from Collection &lt; AbstractTreeNode &gt; to Collection &lt; ? extends AbstractTreeNode &gt; .
Added check for ! f.mViewPagerParticipant &#124; &#124; f.mViewPagerSelected in FragmentManagerImpl &apos;s onContextItemSelected method .
Removed unnecessary &apos; equals &apos; method in RobolectricPackageManager .
Removed instance from instances list after releasing change log lock
Move selected account entry to the top and select it in RSConnectAccountList
Update logger delegate factory class name from io.vertx.logger-delegate-factory-class-name to vertx.logger-delegate-factory-class-name
Incremented revision version from 0 to 1 .
TextField : Changed selectionX calculation to include textOffset .
Update import statements for thrift to wire
Added method seedCreationTimeSecs ( long ) to DeterministicKeyChain , and added parameter passphrase ( String ) to passphrase ( String )
Move isFirstItemVisible ( ) from public to private scope
Replace AtomicInteger with AtomicLong in NodeRelationshipCache
Replace static volatile variable &apos; ourCachedInstance &apos; with final service instance .
Update traffic stats and storage manager code
Changed default value of scan interval from 5000 to 0 and removed auto deploy zipped attribute .
Update ListWithFilter.java to handle Ctrl / Meta + A keyboard shortcut for speed search .
Added support for setup actions in UndertowDeploymentInfoService
Add HBase related configuration to Spark for security mode
Disable fuzzy search in Emmet options
Added LogPosition logPosition ; and synchronized block to ensure atomicity when appending transaction to log .
Added &apos; final &apos; keywords to methods in FileAnalyzerFactory.java
Added beta and GwtIncompatible annotations to Maps.subMap and Sets.subSet.
HmsPicker : Fix for incorrect input when clicking on already selected number
Renamed &apos; ConnectionAbortedException &apos; to &apos; TransportException &apos; and removed implementing interfaces &apos; IsRetryableException &apos; and &apos; IsDeadConnectionException &apos; .
Added &apos; compareTo &apos; method to &apos; AbstractBreadthFirstTraversalTest &apos; class .
Added ignoreInvalidEndpoints ( ) method to RecipientList and MethodInfo classes .
Added check for view opacity and animation in ViewGroup.java
Added import statement for Jvm in DaemonContextBuilder.java
Removed unnecessary System.out.println statement and updated code to use JaxRsResponse instead of Date .
Renamed &apos; trustStore &apos; to &apos; keyStore &apos; in SSLConfig
Update string manager reference from Http11AprProtocol to Http11Processor
Added button with ID &apos; my _ button &apos; to layout file .
Replace FuncN with Func1 in RxPermissions.request method
Fixed Windows-related issue in AccumuloTest by removing isWindows ( ) method and skipping Minicluster setup for Windows .
Added writeObject method to GenericVertex and Vertex classes for serialization .
Added solid color property to GradientDrawable .
Set Z order on top to false
Added download region search text to SearchDialogFragment .
Renamed method fromInternal to fromLenient in InternetDomainName class .
Added isAwaitingAuthorization method to VolatileBuddy class .
Incremented base version of DuplicatesIndex from 20 to 21 .
Modified Windows.java to change the way windows are created .
Remove message Uid locally
Added method to copy meta data from another EncodedImage and added a method to check if all image information has loaded .
Restarting dependency graph builder for conflict resolution
Added constructor for TestResources to allow instantiation with Spock
Added a new condition to check if the materialList is null and the materialName is not null , then load the material from the assetManager using the OgreMaterialKey .
Added wait ( 1000 ) to writerNotify.wait ( ) in AsyncWriteEngine.java to check write conditions every N seconds and prevent possible deadlock .
Renamed keyspace from &apos; LegacySSTableTest &apos; to &apos; Keyspace1 &apos; in LegacySSTableTest.java.
Added function swap ( LValue , LValue ) and changed return type of function rotate ( LValue , LValue ) from int to double .
Fix for # 12345 : Changed randomIntBetween ( 0 , pbr.length ( ) - 1 ) to randomIntBetween ( 0 , pbr.length ( ) - 1 ) / / an offset to the end would be len 0
Added option to use debug log in Compiler
Added test for exception throwing in findAndModify
Added Binder.clearCallingIdentity ( ) and Binder.restoreCallingIdentity ( ) around startActivityFromRecentsInner ( ) in ActivityManagerService.java to ensure correct calling identity is restored after the activity is started .
Added volume setting to sleep timer expired method in PlaybackService .
Updated package-info.java to describe Vert.x instance and event loop threads .
Added debug logging for shutdown process in NetworkInstance
Added fallback logic for missing format in PropertyBuilder
Fix ByteCodeUtils to properly handle Void.class
Updated adaptive learning rate algorithm to use similarity to prior updates instead of time decay factor .
Modified AlarmManagerService.java to change the comparison of IAlarmListener rec and listener.asBinder ( ) in method matches
Removed unused import and static final variable in DiscoveryNode
Added handling for NoSuchLogVersionException in HighlyAvailableGraphDatabase
Added support for Snappy compression algorithm in CompressionStrategyFactory and updated CompressingStoreTest to include Snappy compression tests .
Fix for bug XYZ : Interning symbols with slashes now works correctly .
Added tracing wake lock to ImapFolderPusher
Up Operation : Improved handling of database versions and multichunks .
Added methods for getting user login , name , email , and active status
Added check for Markdown files in stripTrailingWhitespace preference
Fixed spelling mistake in SchemaModifier.java
Added constructor for PGLListener class
Added a new runnable to handle hiding of toast views , and removed unnecessary null check for mNextView .
Added check for null access token before adding it to headers
Modified NodeCmd.java to change &apos; scrub &apos; command to &apos; upgrade sstables &apos; and updated documentation .
Removed unnecessary imports and fields in SessionCreateRequest.java
Update Provides.java to correct typo in method signature .
Added test for Bzip2Encoder &apos;s finish method
Replace &apos; PSurfaceJOGL &apos; with &apos; surface = new PSurfaceJOGL &apos; in PGraphicsOpenGL.createSurface ( ) method .
Added test method updateUsedBytesInTierTest to MasterWorkerInfoTest .
Added &apos; public &apos; modifier to ExpectedException field in SuppressionsLoaderTest .
Added new data segment configuration and updated configuration .
Update master server URL from &apos; master-server.terasology.org &apos; to &apos; meta.terasology.org &apos; in NetworkConfig.java
Added check for null before writing object to exporter in J3MRootOutputCapsule # write method .
Fix for fast scroll thumb pinning during overscroll
Added disabled flag to EhCacheFactoryBean and set it to true by default . Also added setDisabled method .
Added field &apos; mActivityIntent &apos; to ActivityInstrumentationTestCase2.java and set it to null along with &apos; mInitialTouchMode &apos; .
Added WebSocket Server Configuration component and enabled JSR356 Websockets by default .
Added classifier support for apklib generation
Renamed &apos; collectionToPopulate &apos; parameter to &apos; collection BindingSource &apos; in ControllersApi.java
Update getApplication ( ) method in ChatEventStream.java to return &apos; chatServer &apos; instead of &apos; fitcDemo &apos; .
Update classloading configuration to use TcclLookupPrecedence
Added userId check in NotificationManagerService to ensure that notifications are only shown for the current user and not for all users .
Added synchronous channel creation and shutdown in LocalChannelRegistryTest .
Removed unnecessary updateTextSelectionFromMessage call in WebView
Removed unnecessary System.err.println statement in AsyncServletIOTest .
Updated formatDate and formatDateTime methods to use getShortDateFormat and getShortDateTimeFormat instead of getFormat and getLongDateFormat respectively .
Added check for missing latitude and longitude values in GeoDistanceFilter .
Added check for file existence before creating CSV reporter
Changed logging level for starting modal dumb mode from INFO to DEBUG
Added method toString ( ) to Alert class and updated hashCode ( ) method to include alertHeaderText and alertUrl fields .
Modified BrowseFragment.java to use a custom ListRowPresenter and set row heights .
Fixed issue with presence map returning null instead of unavailable presence
Added method intersectRectangles ( Rectangle , Rectangle , Rectangle ) and updated Intersector class .
Added support for heads-up notifications in ExpandableNotificationRow .
Added import of StringPartitioningStrategy and updated partitionId assignment to use it .
Added method getRosterItemCount ( ) and added an Iterator for the roster items in the packet .
Removed unnecessary &apos; input.setInputType ( InputType.TYPE _ CLASS _ TEXT ) ; &apos; line in ShareUtils.java
Added ENOTSOCK error code to ZError class and updated ZMQ class to use it .
Added space to standard mode prefix in Recent.java
Update SocksAuthRequest to throw IllegalArgumentException with corrected error messages for non-ASCII characters in username and password fields .
Added functionality to check / uncheck breakpoint items based on their enabled state .
Added import for DumbAware in NavigationAction.java
Update NewUserPageFragment to use Html.fromHtml instead of Resources.getString for displaying terms of service .
Added support for compiling Java source files in a separate daemon process .
Added hasNext ( ) method and improved key ( ) and element ( ) methods in KeyValueSource class .
Added parameter to use transfers.txt file for GTFS and updated OTPConfigurator to use it .
Modified StoreView.java to gather user-keys that pass all filters and skip first _ offset keys .
Disable HDFS cache for local mini DFS cluster
Increased consumer delay from 100 to 500 in RssPollingConsumerWithFeedHeaderDisabledTest .
Added case sensitivity checks for column names in AndroidDatabaseResults .
Fixed memory leak in CropImageView by recycling viewBitmap only when it &apos;s not the same object as croppedBitmap
Added alpha property to MonthView
Added method headers ( String ... headers ) to set header names and values , and added a check for illegal arguments .
Added warning message for Quasar Java Agent in Fiber class .
Removed unnecessary location flags in PhoneStatusBar.java.
Renamed getPrevNodeOf method to getPrevNodeOfPrivate and made it private .
Added support for importing parameters from a file
Added schema to URL to prevent client crashes
Added log expression and condition enabled flags to XBreakpointBase
Added location-based Wemo Binding
Disabled testGetTableNamesException and added tests for listing unknown schemas and views
Renamed class from &apos; core.ApplicationHttpRequest $ AttributeNamesEnumerator &apos; to &apos; ApplicationHttpRequest $ AttributeNamesEnumerator &apos; in SecurityClassLoad.java
Update StoreDefinitionsMapper to read store list from StringReader
Ignoring known problematic script in AbstractIntegrationTest
Added constructor for Text class with horizontal align parameter .
Added shutdown variable to AbstractActiveService to track whether the service is still running
Update ManagementClient.java to handle exceptions properly .
Added translucent status bar to ActionBarActivity
SvnUtil.java : Removed unnecessary factory creation and simplified content retrieval .
Added userID field to Facebook Connect plugin &apos;s JSON output
Added support for automatically downloading and caching NED data to cover every vertex of a graph .
Added method unsetComputedValue ( ) to GraphTester , and updated setBuilder ( ) to allow setting a SkyFunction builder .
Added import statement for org.gradle.api.logging.LogLevel and updated CodeNarc class to use LogLevel.INFO for capturing standard output .
Removed unused logging statement in VectorMapOperator
Remove unnecessary type parameter in Rule.params ( )
Added getSystemPhoneApplication ( ) method to TelecommManager .
Removed unused import statements in DefaultMavenPom.java.
Removed unnecessary &apos; nextRequest ( ) &apos; calls in Http11AprProcessor .
Added CounterUpdateFilter to server filters .
Added support for asynchronous requests by implementing AsyncCompletionHandlerBase in TypeMapper and updated TestFancyClientBuilder to use Futures for async request handling .
Added support for pathlib.Path.open in PyStdlibTypeProvider .
Removed unused imports and simplified copyFile method .
Allow CharSeq.tabulate to accept nullable function argument .
Added new error message for result type mismatch .
Added long click event listeners for icons and library buttons .
Added a random seed for flush priority to CompactionQueue
Added isPowerOfTwo and nextPowerOfTwo methods to PGL class .
Remove column family and super column in ColumnFamilyStoreTest
Fixed issue with incorrect meta attribute being returned for MySQL enum values .
Added method getOrCreateClass ( String , OClass ) to OSchema interface and removed deprecated method getVersion ( ) .
Update Server.java to use NON _ NULL inclusion in JSON serialization and remove unused imports .
Renamed &apos; deleteMappings &apos; and &apos; deleteAllMappings &apos; methods to &apos; removeMappings &apos; and &apos; removeAllMappings &apos; , respectively , in WireMock class .
Added check for current blog existence before updating filter text in MediaGridFragment .
Update ExpandableItemIndicatorImplAnim to use START instead of END for expanded state .
Remove unused variable AngleCalc and rename AbstractRoutingAlgorithm to RoutingAlgorithm
Updated timeout for response in BasicAuthTest from 30 seconds to 3 seconds .
Change ID.java and IndexingStamp.java to use int instead of long for unique IDs
Added contract for all elasticsearch loggers and updated factory to return instances of ESLogger .
Added cast to AtmosphereResourceImpl.class and call to asynchronousProcessor.invokeInterceptors ( handler.interceptors ( ) , webSocket.resource ( ) ) for proper dispatching of AtmosphereInterceptor .
Remove unused fields and methods in MatchQueryBuilder .
Added documentation for Abstract Factory pattern and updated factory interface and implementations .
Fix EOFException message to not include System.identityHashCode ( this )
Fixed lexer issue with terminating token .
Modified AccessibilityInteractionController.java to correctly handle null parent node in prefetchVirtualDescendantSubtree ( ) method .
Update TemplateResourceReader to use version 2.3.20 of the incompatible improvements .
Added null check for getBodyAs ( Type type ) method in RetrofitError class
Fixed security exception in Clicker.java by removing unnecessary assertion .
Replace declarationToDeclarations.get ( descriptor ) with declarationToDeclarations.get ( descriptor.getOriginal ( ) ) in BindingTraceContext.java.
InferenceContext : Remove unnecessary null check and simplify type inference
Added removeAllListeners method to ApplicationEventMulticaster and AbstractApplicationEventMulticaster , and updated setBeanFactory method to take a BeanFactory argument .
Fix for incorrect language level detection in stub parsing
Fix for array overflow in ASeq.toArray
Added null check for browser before going back
Added debug statement for authentication check in AbstractAjaxCallback
Added missing file indicator and updated path method to AbstractHgTestCase .
Added restriction for restricted users to establish VPNs
Updated code style settings for wrapping fields , variables , and groups
Removed unnecessary code in SourceFormatter.java
Discard statefulSessionComponentInstance before releasing lock and register a tx synchronization for this SFSB instance
Fixed error handling in PifaceBinding by removing unnecessary byte response variable and simplifying retry logic .
Added field &apos; roundRobinIndex &apos; to class Sender
Added check for null issuable before adding issue to file .
Added Contract annotations to Logger class .
Added debug logging for socket status and state in AbstractProtocol .
Removed unnecessary imports and updated jar scanning code .
Fixed incorrect import statement and updated createAttribute method to use correct file type .
Removed unused variable lastReceived and updated subsnitch method .
Modified DBUtils.java to count unwatched episodes without specials .
Added import for Balloon class and updated EventLog to use it for displaying sticky balloons .
Fixing bug in ShellBolt.java where we were trying to write an object to bolt when it was not supported . Now we throw a runtime exception if the object is not of a known type .
Modified MediaRouteButton.java to use selectRouteInt instead of selectRoute for better performance .
Added a space after each address in the toString ( ) method of the Address class for better readability .
Added parentheses to reason phrase in HttpResponseStatus constructor
Added XPath test cases and updated ignore list
Load autosave after first onDraw call to prevent layout breakage
Added TRACE log level
Updated database version to 36 and deleted Simperium DB
Removed unnecessary token parsers and simplified highlighting logic in PlainTextSyntaxHighlighterFactory .
Added import of ZipkinQueryApiV1 in EnableZipkinServer annotation .
Added webappVersion field to StandardContext with null check in setWebappVersion method .
Replace &apos; groups.add ( group.getTextTrim ( ) ) ; &apos; with &apos; groups.add ( group.getText ( ) .trim ( ) ) ; &apos; in Roster.java
Added deprecation warnings for ConflictsDialog constructors .
Modified UriBeaconDiscoveryService.java to create a pending intent for navigating to the beacon &apos;s URL .
Added handling of UNTRUSTED and COMPROMISED trust levels in AxolotlService .
Update getVirtualFileByName method to use refreshAndFindFileByPath instead of findFileByPath
Update hashCode ( ) implementation in FieldAccessor to use 31 as multiplier and fix constructor mismatch in apply ( ) method .
Update LayoutShadowNode to use CSSConstants.UNDEFINED for default margin and border widths .
Improved popup window size handling in AbstractPopup .
Modified flag for activity clear top and new task
Update MethodWithGenerics.java to use Entry instead of Map.Entry for generic type parameter
Update FileHelper to use FileObject instead of JavaFileObject .
Fixed issue with focus gain in LanguageConsoleImpl
Added volatile modifier to AssertingListener fields
GradleTaskManager : trim vmOptions before passing it to BuildLauncher
Updated HttpResolution.java to use raw values from location header and include query in URL .
Fixing callback invocation in SingleChannelAsyncMongoConnector and MongoAsynchronousSocketChannelGateway
Update copyright year in FileSystemWatcherTests.java and add synchronized list for changes .
Modified MessageListFragment to use Map.Entry instead of iterating through keys and values separately .
Increased exclusive file access manager timeout values .
ShellWidget.java : Don &apos;t capture keys when a completion popup is visible and simulate pageup when user hits Page-Up from inside the console input .
Reset connections for disconnected mobile data state
Fixed issue with transaction synchronization in HibernateJpaDialect.java
Added method onError for printing received errors and overridden onUserPrivMessage to handle user private messages .
Fixed bug in HttpMessageDecoder.java : removed unnecessary checkpoint ( State.READ _ CHUNK _ FOOTER ) ; and return null ; statement .
GELFProcessor : Skip underscore in message field key and avoid duplicating short _ message and host fields .
Added check for image ! = null before loading cover bitmap
Added implicit output for android aar bundle in BazelAndroidLibraryRule .
Added support for RubyMine in FlatWelcomeFrameProvider
Added Beta import and updated ConditionBlock annotation with Beta retention policy .
Fixed intercept adjustment for GLM model
Modified Build.java to load properties from a different file location .
Added cancellation support to Promise class
Added log statement to FileDataModel to track number of lines read
Added default value for mappedName ( ) in Resource annotation .
Added import statement for jodd.util.ReflectUtil and updated PropertyResolver to skip over Collection classes .
Fixed print statement in TestJoystick.java
Added a default schema comparison for empty schemaComparison arrays
Added methods for converting CompletionResult to and from JSON .
Added one historic activity instance and one historic task instance to FullHistoryTest .
Futures.java : Add missing Executor parameter to addCallback method
Added error message prefix to assertion messages in BaseCheckTestCase .
Added minimum size constraint to TransferCallDialog
Removed unused import statement for MISSING _ ATTRIBUTE
MathUtils : Changed randomBoolean method to return a random value between 0 and 1 , and removed unnecessary code .
Added import statement for org.junit.Rule
Removed unnecessary print statement in JSONDirectTest _ private.java
Changed default background color of confirm credentials screen from gray to teal ( # 00796B ) .
Fix NfcAdapter.java : Remove unnecessary method arguments and improve method documentation .
Modified DefaultInterceptorsTestCase to use MethodInterceptor.MESSAGE + message instead of Assert.assertEquals ( &quot; Hello &quot; , MethodInterceptor.MESSAGE + message ) ; and added MethodInterceptor . MESSAGE = &quot; MethodInterceptor &quot; ;
Robolectric : Ignore errors when binding shadow classes for Google Maps APIs
Sent message &apos; { } &apos; with attachment &apos; { } &apos; to &apos; { } &apos; .
Fixed issue with JAXRDependencyProcessor.addDepdenency ( ) method . Now , it correctly sets the &apos; optional &apos; flag to &apos; false &apos; for system dependencies .
Added logging statement for databases directory in OServer constructor
Fixed issue with targetRef not being properly set in BpmnParse .
Fixed shader compilation error by formatting shader source correctly .
Added unlockCanvasAndPost ( ) call in stopDraw ( ) method to improve performance .
Removed unused code in NewChatActivity.java
Update AuthenticatorActivity to use mAuthStatusIcon and mAuthStatusText instead of mServerStatusIcon and mServerStatusText for displaying authentication status .
Modified PermuteArgumentsFix to use CodeInsightUtil and TypeConversionUtil instead of PsiUtil and IncorrectOperationException .
Allow generated replies in messaging notifications .
Removed unnecessary network updates registration
Fixed issue with voice prompt suppression when coming back from U-turn .
Removed parameter &apos; isUnicodeDefault &apos; from PyStringLiteralLexer constructor .
Added support for creating bitmaps from streams using Robolectric &apos;s `newInstanceOf` method .
Added import for DaemonAnalyzerTestCase in GoImportOptimizerTest.java
Added validation for disk threshold settings and changed data type of disk threshold enabled setting to boolean .
Added support for WUNDERGROUND provider in Precipitation model
Added test for multiline string literal with semicolons in PostgreSQL SqlScriptSmallTest .
Inverted logic in CharMatcher.matchesNoneOf ( sequence )
Added ImageNotFoundException handling when pulling images in TaskRunner .
Added import statement for ImageView and setScaleType ( ImageView.ScaleType.CENTER ) in StatusBarIconView.java
Added method ignore ( Method ) and deprecated ignore ( Member ) in NullPointerTester . Removed unnecessary constructor and updated Javadoc .
Removed unnecessary import statement for &apos; Tester &apos; in BlockServiceHandlerIntegrationTest.java
Added TestDirectory rule for dumping data about failing build
Added imports for UploadFileVec , Log , and Utils , and updated NanoHTTPD class to use them .
Added a &apos; continue &apos; statement in the future.isDone ( ) &amp; &amp; future.isCancelled ( ) check to skip over incomplete or cancelled futures , and updated the logging message to include the number of open channels and idle channels closed in two loops .
Added check for duplicate init parameter definitions in FilterDef and ServletDef classes .
Added import statements for Callable and Java &apos;s standard library , and modified CacheStats.java to include stats for cache lookups and distinguish between manual removals and evictions .
Added guard clause to prevent null pointer exception in Logger.log method
Added test for canceling SimpleFuture and clearing callback
Modified ActivityLauncher.java to always add preview parameter to URL for unpublished posts .
Throw AssetLoadException instead of IllegalStateException when no loader is registered for a type .
Update WPWebViewActivity.java for opening external WordPress links in a webview .
Cancel tethered notification as user .
Removed unused merge status constants
Disables the resource return for the JEDIS subscriber
Added check for file existence before processing annotations in K2JVMCompileMojo .
Modified TaskStackLayoutAlgorithm.java to conditionally include header bar height in initial bottom offset calculation when there is a docked task .
Update import statements and remove unused imports
Added support for chunked transfer encoding in NanoHTTPD .
Update DashboardWidget.java to use TimeRange instead of timeranges.TimeRange and remove unnecessary import statement .
Removed toast message in MyActivity.java
Added HTTPMethod.PATCH to enum
Fixed issue with GetConnectionTimeoutException handling in DruidDataSource.java.
Added instrumentation for intercepted methods with non-static this parameters
Added isOpaque ( ) method with caching support .
Added JRubyUtilLibrary and JavaEmbedUtils imports to JRubyHookDefinition.java
Modified InternalNioInputBuffer.java to fix issue with header buffer size limitation .
Remove unused imports and add missing imports for ObjectDeserializer
Added OPTIONS method to token endpoint for CORS support .
Modified PasswordUnlockScreen.java to change input type for password entry field from TYPE _ TEXT _ VARIATION _ PASSWORD to TYPE _ CLASS _ TEXT &#124; TYPE _ TEXT _ VARIATION _ PASSWORD
Remove unnecessary constructor and log statement in HibernateJpaAutoConfiguration .
Added logging statement for running single tests with pattern
GitInit : Changed import from &apos; git4idea.commands.Git &apos; to &apos; git4idea . GitVcs &apos; and added virtual file parameter to FileChooser.chooseFile method .
Added exception for kafka-spout in JavaCheckstyle .
Rename GrMethodCallExpression to GrMethodCall and update corresponding method signature in PsiUtil .
Added RequestFacadeAspect to TomcatPlugin weaving
Fixed bug in Glide.java where modelLoader was not being set properly .
Fixed issue with refreshObject in DataSourceDescriptor and GenericDataSource .
Remove deprecated URL getter and use url ( ) method instead .
Added write action for VirtualFileManager refresh to avoid race conditions .
Added import statement for Set and modified batching logic to use Maps instead of Lists .
Added synchronization mechanism to InProcessChronicleSource
Added parameter to WebChromeClient.onSelectionStart ( ) and WebView.selectionDone ( )
Remove unnecessary empty block check
Removed unused import statements for Base64 and UUID .
Added a check to suppress turn prompts when calculating a route and updated voice router status accordingly .
Implemented logic for selecting top N rows in Presto operator .
Added check for connected devices in BluetoothControllerImpl
Fixes issue # 12345 : Changed getUserStateLocked ( callingUid ) to getUserStateLocked ( resolvedUserId ) in TvInputManagerService.java
Changed application context from &apos; context.getApplicationContext ( ) &apos; to &apos; appContext = context.getApplicationContext ( ) ; &apos; in DrawerMenuAdapter.java
Added logging statement for case where no contact is found in ServerStoredContactListYahooImpl .
Added implementation of CharSequence interface to ShadowSpannableStringBuilder .
Fixed issue with skipping blocks starting on boundary
Added fade scroll bars and scrollbars on top option to ScrollPane .
Removed unused code in ReaderPostListFragment .
Renamed constructor parameter from &apos; s &apos; to &apos; message &apos; and added a new constructor with a single string parameter &apos; field &apos; .
Updated SpringSecurityCoreVersion to handle incorrect spring versions
Update Analytics tracking for Reader Comments
Added @ RemotableViewMethod annotation to TextView.setEnabled ( boolean ) method .
Added support for Font Awesome icons in PlaygroundActivity .
Move method addLeadingZero from CommonUtils to test utils
Removed unnecessary null checks for Principal objects in JbossIntegrationFilter .
Added drag start event handling for ViewGroup children
Modified LiquibaseGenerateChangeLogMojo.java to use new DiffOutputControl ( outputDefaultCatalog , outputDefaultSchema , true ) instead of DiffOutputControl ( )
Modified MacMessagesImpl.java to set key equivalent for alert buttons .
Added import statement for IOBalancer in TcpIpConnectionManager.java
Added PGPDataFormatTest for signing and encrypting messages
Add SonarQube copyright and license headers to DaoComponent.java.
Reformatted Java code to improve readability . Removed unnecessary boolean variable and simplified logic for logging . Also , fixed incorrect indentation in some places .
Added retry mechanism for block locking to handle intermittent failures
Added Liquibase.SHOULD _ RUN _ SYSTEM _ PROPERTY check in SpringLiquibase.afterPropertiesSet ( ) method .
Fixed crash when media is null in FeedItemMenuHandler.onPrepareMenu by adding a check for null media before calling getFeedMedia ( ) and updated FeedManager.getInstance ( ) to use the correct media ID .
Added MapMaker for caching directory scans in TreeSnapshotter .
Added backgrounding support to ProgressWindow .
Fixed comparison logic in DanmakuUtils.java
Changed gradient color calculation to use Math.max ( ) function to prevent negative values .
Added method posToString ( ) and updated toString ( ) method to include playback position .
Added encryption status check using LockPatternUtils .
Fixed issue with clearing task stack in SearchDialog
Added exception handling in CommandResultCallback
Added support for getting Android resource path from system property and improved error handling .
Fixed issue with SQLEditor maximized control
Update WELCOME _ MESSAGE _ URL from http to https .
Added check for XML file extension before loading properties from XML file .
Added vertex and edge ID factories to IdGraph
Fix incorrect unit test for DiskThresholdDecider
Added PropertyPlaceholderAutoConfiguration to JolokiaAutoConfigurationTests
Added functionality to invoke channel listener when complete flag is set in StreamSourceFrameChannel .
Modified FrameMap.java : updated enterTemp ( ) method to use enterTemp ( int size ) instead of myMaxIndex + + .
Update MapELResolver to use string literal for null value
Added check for requestStartedTime before calculating duration in SQLBaseResponse .
Update DynamicEndpointSnitch to use double division instead of integer division for calculating penalties .
Update copyright year and add @ SafeVarargs annotation to enableFor method .
Added initialize method to Service interface
Update BoolQueryParser to use boost factor for must / should queries
Added getBiteChance and setBiteChance methods to Fish interface .
Implemented immediate dim animation for launched tasks from app with thumbnail
TaskTest : updated link URL from &apos; https : / / www.url.com &apos; to &apos; https : / / ww.url.com &apos; .
Added USER _ MANAGER and DEVICE _ POLICY _ MANAGER exclusions for Motorola devices running KitKat
Added method removeAllMenuButtons ( ) to FloatingActionMenu class .
Added getCustomListTags ( ) method to ReaderTagTable and updated ReaderUpdateService to use it .
Removed unnecessary &apos; outputBuffer.checkConverter ( ) &apos; call in Response.java
Added logging for failed remote requests
Fixed typo in AntAnnotator.java
Added AmazonIapManager constructor and removed static getter method returning null .
Fixed issue with InsertGenerator where empty strings were not being properly quoted .
Added stack trace printing for debugging purposes in UrlNetworkManager .
Dispose of focus trackback properly
Added check for isUpToDate ( ) call from Event Dispatch Thread to prevent deadlocks
Fix issue with private presenter visibility
Removed unnecessary charset check in NettyAsyncHttpProvider .
Updated intent filter for custom app to handle calendar events
Remove unnecessary dismissColorFade ( ) call in DisplayPowerController .
MongoFind.java : Implemented order ( ) method and removed unsupported hintIndex ( ) method .
Added super.loadState ( element ) in DependencyValidationManagerImpl to fire listeners after all reads .
Added break statement in DownloadObserver to improve readability and fix issue with infinite loop .
Update StatisticalCorefProperties.java to use marked discourse for coreference resolution when using a rule-based mention detection model .
Revert DEBUG flag to true for PatternUnlockScreen before JB release
Added generic type parameter to PathRepresentation constructor .
Added Mesos integration for Alluxio
Added syslog-redirector to container entrypoint and copied image cmd if not specified
Update builder className calculation to use fully qualified name instead of simple name .
Added enableTextureButtons ( ) call to TerrainEditorTopComponent &apos;s addRow method
Added check for null before iterating over command map in AceEditorIdleMonitor
Fixed bug in HtmlDocumentationProvider where isTag was incorrectly determined when element &apos;s parent was an XmlAttribute
Added --max-res-version option to AndrolibResources.java and updated usage ( ) method in Main.java
Added import statement for StatementLocksFactorySelector and added a new statement locks factory .
Added fallback for densityDpi calculation when DisplayMetrics.densityDpi is not available
Added logic to handle missing rise and set times for moon calc
Replace WorkerPool with RingBuffer and YieldingWaitStrategy
Added file lock disable and file mmap enable for crash test
SubscribedSession.sendAnyPendingSMs ( ) - &gt; SubscribedSession.sendAnyPendingSm ( )
Added print statement for installed plugins
Added method setActiveProfileOwner to register a component as an active admin and profile owner for a user .
Update encryption algorithm to use ConcealEncryption for highest security level
Added logging statements to Daemon class
ZAP : 2015 / 02 / 02 Move output panel help key registration to prevent NPE
Modified InstanceCreationUtils.java to use VIEW _ DELEGATE _ CONSTRUCTOR _ SIGNATURE instead of null for the constructor signature .
Fixed issue with sending network state change broadcast and clearing IP settings when disconnecting from a Wi-Fi network .
Renamed JetPackageDirective to implement JetExpression instead of JetReferenceExpression
Remove unused import statement for Preconditions
DetailsPanel.java : Added &apos; committed by &apos; information to author text
Added check for disabled state before adding comment to request arguments .
Set isDirty flag to true when marking RecordsTable dirty .
Removed unnecessary line of code in DrawerBuilder.java
RestRequest : Changed int parameter parsing to long parameter parsing .
VimShortcutKeyAction : added check for insert mode and file editor before allowing escape , tab , or enter keys .
Replace &apos; getName ( ) &apos; with &apos; getText ( ) &apos; in getSelectedColorName ( ) method .
Added check for null return value and Void.class return type in AbstractMessageConverterMethodProcessor .
Added test for GroupProperty and GroupProperties classes with HazelcastSerialClassRunner
Added logging for Jingle Nodes relay discovery termination and updated relay entry count for account : &lt; accountID.getAccountAddress ( ) &gt;
Fix for BluetoothGattCharacteristic.getStringValue ( ) method to handle null value .
Added methods isActionDown ( ) , isActionUp ( ) , isActionMove ( ) , isActionCancel ( ) , and isActionOutside ( ) to TouchEvent class .
Update reference from internal AuthenticationSpecification to specification AuthenticationSpecification
Added support for detecting direction from player &apos;s input , and updated handling of unknown directions .
Added assertion for ExecTask in TaskViewServiceTest
Added support for custom file patterns in FindInFilesDialog .
Update InfinispanCacheStoreManager to correctly reflect Infinispan &apos;s distributed mode capabilities .
Fixed typos in ListActivity.java
Increased local Tachyon cluster resource size from MB to GB
Added method putJSONAA4 to AutoBuffer class .
Added check for remaining data in WebSockets.sendData ( ) method
Added support for assurance contracts in WalletProtobufSerializer
Renamed LdapResourceCRLFetcher to LdaptiveResourceCRLFetcher and updated attribute name in AbstractX509LdapTests .
Added exclusivity checker to LoadBalancingProxyClient to honor Upgrade requests
Fixed OutOfMemoryDialog to correctly display the path of the write file when saving data .
Update SearchHealthMediumTest to use greater than or equal to instead of just greater than for testing JVmThreads and open files .
Removed unnecessary print statements in ModelFactory.convertProperty method .
Increment cache layout version in DefaultCacheLockingManager from 13 to 14 .
Added Languagetool as a suggestion for the word &apos; Languagetool &apos; in the spelling checker .
Added action listeners for validate scope controls and find button in FindDialog .
Renamed &apos; AppCallbackHandler &apos; to &apos; AppCallbackHandler &apos; and removed unnecessary &apos; public &apos; modifier .
Added check for vertex execution state in MulticastManager.java
Removed unnecessary import statement for com.apple.eawt.AppEvent
Added check for generator settings before creating world generator
Modified Parse.java to improve error handling when parsing CSV data .
Added getIndexedString method to GotoRelatedFileAction to provide custom name for related items
Added createShardRewriter method and updated OrcStorageManager to use it
Fixed issue with lock pattern timeout
GhostView : Remove unnecessary invalidate ( ) call and use setGhostedVisibility ( View.INVISIBLE ) instead .
Set openOnTapEnabled field to &apos; openOnTapEnabled &apos; instead of &apos; _ openOnTapEnabled &apos; for better naming consistency .
Replace ExpressionReplacingVisitorSupport with StatementReplacingVisitorSupport and remove unused variable .
Added hashCode and toString methods to AntPathRequestMatcher
Added logic to handle screen state when dozing and block screen on
Wizard dialog visibility fixed .
Added import statements for Entity and LivingEntity , and added getPlayers ( ) method to World interface .
Added check for public method of non-public class and updated exception message
Removed testPostWithContinuation and updated testSendAck in TestHttp2Section _ 8 _ 1.java
Added a new TextView to getSuggestionItemHeight to improve efficiency
Replace getPrivateDnsNames with getPrivateIpAddresses in AWSClient and TcpIpJoinerOverAWS
Added constructor for HtmlDocumentImpl that takes an IElementType parameter .
Added Config.DEBUG check for enabling consistency check in ViewDebug
Added getStatistics method to InMemorySessionManager and implemented SessionManagerStatistics interface .
Added constructor to Matrix _ Delegate and added test cases for class replacement .
Added fallback isolation semaphore max concurrent requests to HystrixCommandTimeoutConcurrencyTesting .
Update WXScrollView.java to calculate parent height correctly .
Added import of Version class and updated User-Agent header pattern in CallTest.java
Implemented Builder pattern for Hero object construction .
Added assertion to ensure variable declaration is not null in ExpressionCodegen # putLocalVariableIntoFrameMap
Update NodeCmd.java to use correct spelling of &apos; Effective-Ownership &apos; in output format string .
Added receiver to dead receivers list .
Removed unnecessary log statements in IccCard class .
Added Logger.getLogger ( &quot; net.bytebuddy &quot; ) .info ( &quot; Only running agent application with a single type locator on Travis CI server &quot; ) ; to AgentBuilderDefaultApplicationTest.java
Update Flickr API URLs from http to https .
Added switch statement to handle different types of enclosing elements for ProvidesProcessor
Added import statement for ContainerUtil in ChangeListWorker.java
Fixed issue with implicit lock interval in IndexerCoordinatorResource
Fixed issue with tracking branch detection in CurrentBranchDialog .
Fix issue with non-persistent connections in EarlyCloseServlet .
Increased max intermediate rows limit in GroupByQueryEngine
Fixed JavaScript timers not being resumed when app is backgrounded
Update bluetooth status bar policy to show &apos; quick settings bluetooth on &apos; instead of &apos; bluetooth disconnected &apos; when bluetooth is enabled and connected .
Added @ Override annotation to JedisSlotBasedConnectionHandler.getConnection ( ) method .
Fixed bug in IQBindHandler where resource was not being set properly .
ZLTextView : Show scrollbar when it &apos;s either shown or shown as progress .
Fixed ClassResolver to properly handle ClassNotFoundException and removed unnecessary NoClassDefFoundError catch block .
Removed unnecessary print statements in EvictionHelper.java.
Added import for com.intellij.openapi.project.DumbAware ; and implemented DumbAware interface in ComboEditorCompletionContributor class .
Added import statement for &apos; java.util.concurrent.RejectedExecutionException &apos; and updated the expected exception in the &apos; testExecute _ withNoMemberSelected &apos; method .
Added system store to remote metadata updates and added getSystemStoreDef method to fetch system stores .
Added UIScrollbar to UIDropdown and updated UIDropdownScroll to use it .
Deleted extractor &lt; extractor _ id &gt; of type &lt; extractor _ type &gt; from input &lt; input _ id &gt; . Reason : REST request .
Fix for IDEA-223448 : Remove modal dialogs in PrepareToDeployAction and SvnEntriesFileListener
Update GCM key documentation
Fixed optimization issue in ATNOptimizer by removing unnecessary cast .
Added check for &apos; myDisposed &apos; in LivePreviewController &apos;s updateSearchResults method to prevent NPE when the component is disposed .
Replace import of Player from SpongePowered to Forge for PermissionsProvider
Unregister network factory when unvalidated networks are reaped
Added SwingUtilities import and updated showProgress ( ) method to use invokeLater ( ) for better performance .
Added import statement for &apos; CmdLineException &apos; and updated the catch block to handle this exception .
Added mapping for bmp images
Removed unnecessary import statement for &apos; org.joda.time.DateTime &apos; and updated the creation of TagDefinition objects to use the &apos; UUID.randomUUID ( ) &apos; method instead of &apos; new DateTime ( ) &apos; .
Added override of toString ( ) method in SkylarkNestedSet class .
Removed unused import : org.apache.abdera.parser.ParseException
ExceptionAction enum and Invocation class changes
Fixed issue with liquid filling in BlockTank.java
Increased indent width from 15 to 20 in DraggableTaskListActivity .
Disable JDWP transport in JpsServerManager .
Update NetworkController.java to remove unnecessary log statements and simplify code .
Added Apache License version 2.0 to OttoBus.java
Added implementation for getCollectionManagerClass ( ) method in ODistributedStorage class .
Fix URL encoding in ServerRestInterfaceRealm login
Added PyOverrideTest to PythonAllTestsSuite
Change journal entry type from ADD _ MOUNTPOINT to DELETE _ MOUNTPOINT in DeleteMountPointEntry class .
Added printStacktrace method call to handle heavy filters in AnalyzeStacktraceUtil .
Fixed Vector3.nor ( ) method to properly handle zero-length vectors .
Removed unused import statements and methods in HibernateSession.java
Added final modifier to LocalTachyonCluster field in TachyonFSContract
Added exemption for PinCodeActivity and set pass code timeout to 1000 . Also , added flag Intent.FLAG _ ACTIVITY _ REORDER _ TO _ FRONT when starting PinCodeActivity .
Changed action button text color in FileDisplayActivity .
Added feature for PSEQ with DistSim
Added try-catch block for fileWatcher.startup ( ) to handle IOExceptions and removed unnecessary fileWatchNotifier.addNotificationListener ( defaultListener ) ; call .
Modified SourceFormatter.java to use ContentUtil instead of FileUtil for reading portal.properties file .
Updated Main.java to use LinkedHashSet instead of HashSet for commandParams .
Added @ Support and @ Transition annotations to DSL.java
Remove unsupported method testReadOnlyNearCacheConfigSetCacheLocalEntries and add test for setInMemoryFormat with string
Removed deprecated PyElementType DECORATED _ FUNCTION _ DECLARATION and added new PyElementType ARGUMENT _ LIST , IMPORT _ ELEMENT , and STAR _ IMPORT _ ELEMENT .
Deprecated NinePatchDrawable constructor and added new constructor with Rect parameter .
Added ProgressManager check in BunchFactory
Added method fileChannel ( ) to return the FileChannel to the raw log of the Image .
Removed unnecessary &apos; classVisited &apos; field and optimized visitor logic .
Added constructor for ShadowHttpGet with URI parameter and updated tests .
Added field &apos; secondElement &apos; to class EventContentFilter and updated constructor to take an additional parameter .
Modified ProviderTestCase2.java to extend MockContext instead of Context
Reset evalWaitInterval only if new route is in forward direction
Added constructor for ClassFilterEditor with ClassFilter parameter .
Catching Exception instead of XMPPException in XmppConnection.java
Removed unnecessary copy constructor in DefaultImmutableDirectedAcyclicGraph .
Added variable &apos; edge &apos; to calculate the correct top position for dragged items in DragSortListView .
Added French localization for DateCheckFilter .
Added port ( ) method to BaseActualServer and implemented Server interface in BaseServer .
Added test for PreStartupStoreUpgrader
Added user identity splitting to WebApplicationHeader
Fixed race condition in CompositeElement.createAllChildrenPsiIfNecessary ( ) by iterating over children instead of using acceptTree ( ) method .
Added support for forcing dependency version changes
Added support for filtering stream messages by creation time .
Added import statement for Map ; in TransportClientNodesService.java
Added isOccluded method to StatusBarKeyguardViewManager and updated PhoneStatusBar to use it .
Added support for multiple camcorder profiles
Added binding for DefaultLocalClientChannelFactory and DefaultLocalServerChannelFactory in NettyModule .
Suppress some internal errors caused by bugs in specific JDI implementations .
Modified UriEndpoint annotation for ConsulEndpoint to use &apos; consul &apos; scheme and updated label to include &apos; cloud &apos; .
Fix for bug # 12345 : Remove extra loop iteration in NioServerSocketPipelineSink
Added check for position of interest in InitialInfoBuilder
Removed unused import and field for KeyguardManager
Replace MultiEnumeration with a single Enumeration .
Modified GlobalAuthenticationPolicyWebflowEventResolver to only check for authentication when determining event for principal , and improved logging messages .
Added user tracing with TraceType.USER and added debug logs for trace closure .
Renamed method doubleToSortableLong to sortableLongToDouble and swapped parameters in FloatingPointBitsConverterUtil
Added catch block for InternalException in DebugProcessImpl
Fixed TTL handling in CassandraDatastore by using the configured datapoint TTL when it &apos;s 0 .
Added fuzzy search support for multiple keys in IndexNodeFunctionalTest
Renamed ShutdownHookThread constructor and added a new argument &apos; name &apos; .
Update WindowManagerImpl to set extended state of frame .
Added missing &apos; , &apos; in @ SuppressWarnings annotation in ErrorHandlingTest.java
Updated LocalRegionCache to use Integer.MAX _ VALUE for max-size config on map .
Added documentation and analysis for ClonesArguments.java
Removed @ Deprecated annotations and added @ SuppressWarnings ( &quot; dep-ann &quot; ) to JspContext.getExpressionEvaluator ( ) and JspContext.getVariableResolver ( ) in JspContext.java , and removed @ Deprecated annotation from JspException.getRootCause ( ) in JspException.java.
Added support for adding a new Python SDK and updating the SDK list .
Remove unsupported add methods from RadixTree and TrieMap
Refactoring : Introduce method duplicates handler
Added check for active tab count before switching to bubble view
Added true positives and true negatives to ConfusionRuleEvaluator output
Removed parameter &apos; fileSystem &apos; from TxLog constructor .
Renamed &apos; PlannerError &apos; class to &apos; APIResponseElement &apos; and changed exception handling for error messages .
IdeaTitledBorder : Increased text inset horizontal spacing from -15 to -20 .
Fixes issue with player interaction manager not properly handling event cancellation and block activation .
Renamed method deleteFromRelationshipIndexnoValue to deleteFromRelationshipIndexNoValue in class RestfulGraphDatabase .
Fixed zoom calculation for vector map rendering
Removed unnecessary System.err print statement in DistributedStoreManagerTest .
Added logging for ignored columns in Job class .
Added prewarm service check for camera intent in KeyguardBottomAreaView .
Added WildcardConfigPatternMatcher to InstancePermission
Made multiResolveImportElement and resolveFromImportStatementSource private .
Added method setVisible ( boolean ) to NotificationCustomViewWrapper and NotificationViewWrapper , which animates the view &apos;s alpha value when visible changes .
Added import statement for BroadcasterCache and added code to add messages to the cache in XHRTransport .
Added code to set tick tuple interval based on topology timeout
Removed unnecessary print statement in CasAuthenticationToken.java.
Added import statement for org.jetbrains.annotations.NotNull ; and updated getName ( ) and getDescription ( ) methods to use @ NotNull annotation .
Added FML marker to ModLoader
Added line of code to set tag to null in BottomBarBadge.java
TreeWalker.java : Simplified comparison of AstNode positions .
Remove unnecessary comment in Tokenizer.java
Added documentation for Iterables.getFirst ( ) method recommending use of iterator ( ) .next ( ) instead when no default value is desired .
Made VfsImplUtil.getHandler public for Kotlin
Throw ApiException ( ApiException.Type.INTERNAL _ ERROR ) instead of ApiException ( ApiException.Type.URL _ NOT _ FOUND ) when StructuralNode is null
Added getSubsystemXsdPath and getSubsystemTemplatePaths methods to SarSubsystemTestCase .
Added getGarbageCollectionStatsWithEmptyDefault method to return stats for unknown memory pools .
Added &apos; pattern &apos; parameter to method getHighestTerrainBlock and updated recurseDirections array .
Added import statement for com.facebook.buck.util.Paths ; in AndroidResourceRuleTest.java
Fixed comparison of EdgeEntry weights to handle NaN and -0
Added EXTRA _ APPEND _ LOCATION and updated HISTORY _ PROJECTION in Browser.java
Updated SignInFragment to switch to dot org icon based on showDotOrg flag
Update quicksort algorithm to have a better worst-case time complexity by using three-way partition and handling equal keys .
Added method toCharArrayForSpringWebSocket to SerializeWriter class
Added wsum accumulation in reduce method of FrameUtils
Added check for Shiny doc in renderRmd method of TextEditingTarget.java
Added import statements for TestRequest and TestSerializablePrincipal , and added TestRequest and TestTomcat test suites .
Simplify jar classpath snapshot factory
Renamed &apos; websites &apos; to &apos; urls &apos; in ContactAccessorSdk5.java
Update dependencies : added JUnit 4.11 and updated Spock 0.7-groovy-2.0 to 0.7-groovy-2.0
Fixed PyExecuteFileLineMarkerProvider to use &apos; file &apos; instead of &apos; element &apos; for getting the containing file .
Added test for undoing drop constraint
Added timeout handling to Peer class .
Added SensorManager binding and propagated context to background thread
Increased time for mock endpoints satisfaction in XmlJsonStressTest from 30 seconds to 60 seconds .
Added @ Ignore annotation to testBulkSetGet _ withManyKeys due to issue # 9637
Fix grammar mistake in RecyclerView.java
Fixed bug in Node.java by removing null check for id and adding missing semicolon at end of file .
Added new GBM tests for alphabetical categorical features and Bernoulli distribution .
Added optional JSONArray parsing for views in DynamicView
Renamed test discovery configuration name to include module name .
Renamed Study project to Learn Python
Removed unused variable MEMBER _ LIST _ PUBLISH _ INTERVAL _ SECONDS
Added code to set keyboard focus on window when it is drawn
Allow forceEcho to be set for Knitr output
Fixed issue with HiDPI calculation on Windows .
Added method getResponse ( ) to AtmosphereResponse class to handle proxy responses .
Fixed error handling in TransferOnePidgeyExample by logging exception instead of printing stack trace .
Added null checks for files and accounts in isDownloading and isUploading methods .
Added check for return type of property getter method in GroovyPropertyUtils .
Added setting of custom POI filter for map
Added logging statement for lock acquisition failure in TruncateTranslogIT test .
Added isOk method to CommandResult class .
Removed unused variable &apos; totalDone &apos; and simplified &apos; addPendingMessages &apos; method .
Introduce variable : handle array initializer expressions correctly
Removed unused method &apos; forComponent &apos; and simplified &apos; forComponents &apos; method in DefaultArtifactResolutionQuery.java
Added newline character to make the error message more readable
Added import statement for android.view.ViewGroup and updated layout parameters for title and text views based on progress bar visibility .
Removed unnecessary viewport test cases .
Update GrailsUtilTests to use public void testGrailsVersion ( ) { instead of void testGrailsVersion ( )
Modified getField method in MessageProvider to use ContentUri.buildUpon ( ) instead of concatenating strings
Removed unused import statement and removed @ Repeat annotation from test method .
Added import statement for com.intellij.openapi.util.text.StringUtil ; and modified processOutput method to use StringUtil.convertLineSeparators.
MavenTestCase : updated setFileContent method to use WriteAction for better performance
Modified ElasticsearchClient.java to use private static final variables for DEFAULT _ CLUSTER _ NAME , DEFAULT _ INDEX _ KEY , and DEFAULT _ REMOTE _ HOST instead of public static final variables .
Added premultipliedAlpha field to ParticleEmitter
Fix grammar to avoid potential underflow in IntStream.java
Removed unnecessary &apos; serialVersionUID &apos; field in StreamComparator class .
Added block transfer functionality and updated license header .
Fixed parameter table model receiver in JetChangeSignatureDialog
Added log statement for debugging purposes in CentralAuthenticationServiceImpl
Added import statements for &apos; java.nio.channels.Channels &apos; and &apos; java.nio.channels.ReadableByteChannel &apos; and updated code to use &apos; Channels.newChannel ( socket.getInputStream ( ) ) &apos; instead of &apos; socket.getChannel ( ) &apos; for creating the &apos; NIODataInputStream &apos; .
Update AssetExtractor.java to catch all exceptions instead of only IOException .
Added method &apos; asByteBuffer ( ) &apos; to class ByteString and updated test case ByteStringTest .
Fixed query queue selection to use session instead of statement
Remove unused import statement for Mockito
Added support for finders in resource schema collection .
Update configuration option SYSTEM _ LOG _ TRANSACTIONS to default to false
Added support for rendering literals of types BIGINT , BIT , BOOLEAN , DECIMAL , DOUBLE , FLOAT , INTEGER , NULL , NUMERIC , SMALLINT , and TINYINT in SQLTemplates .
Added override of equals method in EstimatedHistogram class and updated serializer to use ICompactSerializer2
Added method isJavaVersion ( String version ) to TestSupport class .
Removed import of SystemUtil in ProxyInfoTest .
Added getDimensionServiceKey method to GitFetchDialog
Added a new line to set the first element of the arrays to false before iterating over it to skip null references .
Added support for Actor Push notifications .
Modified MessageBrokerBeanDefinitionParser to use WebSocketNamespaceUtils.registerScheduler ( ) instead of hardcoded bean names .
Modified PinotFileUpload.java to change error logging messages and add additional error handling for file uploads .
Removed unnecessary &apos; location &apos; variable in NavigatePointFragment and SearchHistoryFragment .
Added JavaPrint function for JNI debugging .
Remove unused import &apos; java.security.DigestInputStream &apos; in FileHashes.java
Added check for GC / Heap size in Timing.java to prevent timeouts due to stuck threads .
Update HiveSessionImpl to use activeCalls instead of pendingCount .
Modified OrientEdge.java to add missing &apos; public &apos; modifier for convertToDocument ( ) method .
Added nullability annotations to CommandProcessorImpl.java
Added import of projectWithResolvedLibraries in AbstractExternalProjectImportBuilder
Added openSocket variable to breakKeepAliveLoop method in Http11AprProcessor and Http11Processor classes .
Resize pinned stack without preserving windows
Added support for XML conversion with SourceHttpMessageConverter in RestTemplate .
Added depreciation warning exemption to WhitelistWarningsGuard .
Fixed issue with BaseDataReader.java : added check for read availability in non-blocking mode .
Fix JavaDocInfoGenerator to correctly handle non-member elements
Fixed session timeout configuration for users resource .
Fixed issue with handling activity results
Added stopActivityMonitor method and called it in finalize and finishActivity .
Fixed bug in WebappClassLoader.java : correctly retrieve parent thread group
Added dependency on yaml package ( version 2.1.5 ) in DependencyManager .
Added check for WebSocketProtocolStream.class.isAssignableFrom ( webSocketProtocol.getClass ( ) ) before calling onTextStream ( webSocket , reader ) .
Catching Throwable instead of CameraAccessException in FlashlightController # initialize
Added method getBoolean ( String key ) to return the value of a field as a boolean .
Added getServer method to App interface and implemented it in AbstractApp and GraphDatabaseApp classes .
ForgeChunkManager : added check for mods attempting to allocate chunkloading tickets beyond their maximum allowed amount
Removed unused import : org.onebusaway.gtfs.services.calendar.CalendarService
Added import for WeakHashMap in PatchDelegate.java
Removed unused import statement for MetaData and updated RecoverySource class .
Added @ Ignore annotation to ProxyITest.java to ignore test since manual proxy setup is expected on port 8888
Fixes issue # 12345 - Changed mProvider to provider in tearDown method of ProviderTestCase and ProviderTestCase2
Added &apos; setIsRootNamespace ( true ) &apos; to contentViewLocal , bigContentViewLocal and publicViewLocal in BaseStatusBar.java
Added support for new theme tags : holiday , photoblogging , seasonal .
Added index parameter to inflateButton method in NavigationBarInflaterView
Added logic to skip GC warning listeners for concurrent collectors
Added support for unsubscription requests from unknown users in PresenceSubscribeHandler .
Added position dirty flag to SpoutPlayer
Added check for savedInstanceState before configuring MagicLinkUI
Fixed issue with incorrect configurable search results by removing unnecessary check for remove event type .
MortarDemoActivity.java : Set actionBarOwner.setConfig ( null ) in onDestroy ( )
Added support for Android M ( API level 24 ) in AndroidExcludedRefs .
ZParams : Changed &apos; weights &apos; method to accept double &#91; &#93; instead of int &#91; &#93; .
Added check for origin repository in RepositoryManager # getRepositoryModel
Added support for real languages , removed demo language
Allow plugins to handle link clicks and cancel navigation
Fixed buffer size calculations for XAttrUtil .
Added import statement for &apos; com.intellij.openapi.project.DumbAwareAction &apos; and modified &apos; GenerateAction &apos; class to extend &apos; DumbAwareAction &apos; instead of &apos; AnAction &apos; .
Added support for specifying activation function in RBM constructor .
Added import statements for Pair and List , and implemented getSourcePaths method in GoModuleBuilder .
Added new amenity types to PoiFiltersHelper : internet _ access _ wlan , internet _ access _ wired , internet _ access _ terminal , internet _ access _ public , internet _ access _ service , post _ office , telephone , toilets . Removed emergency _ phone , marketplace from Other category .
Added method ls ( String , boolean ) to TachyonFS class to list files and directories under a given path , and added a recursive parameter to list subdirectories .
Removed unused import &apos; org.apache.camel.converter.stream.CachedOutputStream &apos; in IOHelper.java.
Added loadLibrary calls for RSSupport and rsjni in RenderScript.java
Removed unused import statement for &apos; Commands &apos; in FilesPane.java
Added support for specifying the position of animated GIF frames using the column and row numbers , and fixed width and height for frames . Also , added a check to ensure that the size is set before adding frames .
Added check to ensure that an account is not added to accountsInOrder list twice .
Added &apos; isDumbAware &apos; and &apos; update &apos; methods to CommonActionsPanel
Added hack to extract jnilib to dylib on OSX Java 1.7 +
Added code to prevent surrounding ScrollView from intercepting swipe gestures during notification row layout .
Added initialization of task list metadata in SubtasksListFragment
Removed unnecessary assertion for rails.root initialization parameter .
Lazy initialize SSLSocketFactory in HttpsConnection
ServletException : Changed constructor to use &apos; super &apos; instead of &apos; this &apos; .
Added English Word Repeat Rule and moved it before LongSentenceRule .
Added space between parameter declarations in BitmapRecyclerDefault.java
Modified GitChangesCollector.java to correctly handle files under a parent directory .
Added getGPUImage method to retrieve the GPUImage instance used by this view .
Added support for Chinese Grammatical Structure in DependencyParser .
Added timeout handling for asynchronous connects in NettyAsyncHttpProvider
Added hidden object filtering to SQL completion processor .
Renamed getGithubUserName and getGithubRepositoryName to be public abstract methods .
Remove unnecessary import statements in UriReaderTest and ServerSettingsTest .
ArrayMultApp : reduced array size from 14 to 10 and updated loop accordingly .
Added ActionBar import and updated layout for TraktSyncActivity .
Added @ Ignore annotation to TestClusterNames.java and TestStoreCopy.java to ignore failing tests .
Update ConsistentKeyLocker to use microseconds for timestamping instead of milliseconds
Added getBaseDir ( ) method to RobolectricTestRunner and updated usage of FsFile .
Changed CardinalityAggregatorFactory to extend LeafOnly ValuesSourceAggregatorFactory for better performance .
Added thumbnail strip visibility toggle in ReaderPostAdapter .
Update JsonImporter to use JSONParser instead of XmlParser for tree data import .
ConnectionState : added AtomicReference &lt; TracerDriver &gt; tracer parameter to constructor
Removed unnecessary &apos; return &apos; statement in StandardService.java
RoutingRequestBuilder : setFrom and setTo methods updated to use &apos; setFrom &apos; and &apos; setTo &apos; instead of &apos; from &apos; and &apos; to &apos; respectively .
Remove unnecessary import statement and set PinpointSocketFactory &apos;s message listener to null .
Renamed bpmn files from &apos; org / activiti / cdi / test / bpmn / &apos; to &apos; org / activiti / cdi / test / impl / bpmn / &apos; .
Fixed grammar error in SymbolChecks.java : Rule has no args .
Added Jackson 2.x XML converter tests and updated dependencies .
Added test case for CoGroupOperator : testCoGroupKeyMixing2
Remove edge label type configuration
Enforce calling permission for start / stop LockTaskModeOnCurrent methods in ActivityManagerService .
Fixed issue with authors list in GitCheckinEnvironment
Fix for onRequestSendAccessibilityEventInternal to use internal method
Fix for natural sorting of method annotations in AnnotationDirectoryPool
Fixed error handling in TextView by delaying display of error messages until after first layout pass .
Added space in resource not found message for ConfigFileApplicationListener
Fix for setting spatial reference in Control.cloneForSpatial ( ) method .
Added logic to move focus to visible behind activity when moving home task back
Added cache directory settings to ignore list for daemon restart
Replace request with newRequest in NettyAsyncHttpProvider &apos;s else branch .
Added support for non-optional framework dependencies in the New Project Wizard .
Update contact status in chat panel
Modified TreasuryYieldReducer.java to include average calculation and updated logging statement .
Added IsNamedMatcher for matching named elements with explicit names .
Modified SquidUtilities.java to remove unused import and parameter , and to use File.mkdirs ( ) instead of folder.mkdirs ( ) for creating directories .
Added import statement for java.util.Arrays ; in ReadResponse.java
ReactImageView : updated RoundedCornerPostprocessor to use ScalingUtils.getTransform instead of mScaleType.getTransform
Removed unnecessary imports and updated LeveledManifest class .
Added runtime exception for graph database not started
Added implementation for ActionBar.TabListener interface in FeatureToggles class .
Added health check for green status in NoMasterNodeTests
ZWaveNodeStageAdvancer : Changed NodeStage.DEAD to ZWaveNodeInitStage.DEAD and controller.getTxQueueLength ( ) to controller.getSendQueueLength ( )
Disabled testMultipolygon pending discussion with author ( AMB )
Ignore usb0 down after enabling RNDIS and ignore interface down for WiFi . Add Tethering _ BLUETOOTH .
Added UnknownHostException handling and improved retry logic in AsyncHttpRequest .
Increased double tap timeout from ViewConfiguration.getDoubleTapTimeout ( ) + 100 to ViewConfiguration.getTapTimeout ( ) + 100 in SimpleTwoFingerTapDetector.java.
Added a fix for the InplaceEditor to properly handle mouse events when the editor is not yet shown .
Added assertEquals statement to TextRunnerTest.java to check exit value of process .
Temporarily enable multiple user support by default .
Added debug logging for JDBC 3.0 getParameterType call not supported
Added necessary permissions to remote naming test case .
Added method sayHello ( String ) to HelloBean class .
Added check for invalid volume and improved error handling in OpenExternalDirectoryActivity .
Added support for Apache XML-RPC protocol in PydevConsoleCommunication .
Added ability to set only after date for VcsLogDateFilter
Added ability to run tests in abstract inner classes
Added test for compressing wildcard type
Added DebuggingWrapper.makeDefault ( ) to AdviceTest.java
Fixed logging tag in BinaryHttpResponseHandler.java
Removed unused variable flipDiagonally and removed unnecessary getter method for rotation .
Updated configuration options for stream processing .
Fixed issue with PyModuleDependenciesConfigurable.java , now correctly checking for modifications
Added MODIFIER _ COMPARATOR _ DURATION _ DESCENDING comparator to IModifier interface
Update copyright year to 2013
Added BiMap for fluid blocks
Log level changed from info to debug in ActiveThreadCountService.java.
Replace &apos; Class &lt; ? &gt; &apos; with &apos; Class &apos; in method signature of &apos; body &apos; .
Added parameter &apos; path &apos; to QueryFilter constructor to allow slicing at a specific level ( CF or SuperColumn ) .
Fixed error handling in PeopleManagementActivity when getDotComBlogId ( ) returns null or empty string .
Removed unused variable &apos; url &apos; in DbImportTest .
Added flag Intent.FLAG _ ACTIVITY _ NEW _ TASK to launch intent for task stack view
Fix bug in DataFileChannel.java : handle negative read ( ) returns correctly
Substitute commit message to ignore whitespaces
Remove unnecessary comment line in NetworkBuilder.java
Added handler for sending unavailable presence when user goes offline
Removed unused method showUpwardsFoldToken and updated saveBeforeSourcing to return a boolean value instead of a PrefValue &lt; Boolean &gt; . &apos; ,
Added onViewAdded method to RCTCameraView to properly handle view finder addition and removal .
Update Builder class to final
Added copyright and license information to DownloadListener.java
Fixed dispose method in VcsLogTabsWatcher to properly remove listeners when tool window is closed .
Added thread-safety for parallel execution in DependencyResolutionLogger
Added sign for contact separately in JsBindingModule
Modified BptBuilderBlueprint.java to change the check for fluid equivalence in crafting recipes from using FluidContainerRegistry.getFluidForFilledItem ( invStk ) to StackHelper.isMatchingItem ( reqStk , invStk , true , true ) . This allows for more flexible and accurate checking of fluid equivalence .
Added logging statements for node upgrade process in UpgradeTest .
Remove filter parse element in favor of post _ filter and postFilter
Update FileAppenderFactory.java to use abbreviations for size units ( e.g. , MB , GB , TB ) instead of spelling out the unit names ( e.g. , megabytes , gigabytes , terabytes ) .
Update JvmComponentPlugin to use binary naming scheme for API classes directory .
Removed unnecessary import statements and debug logs .
Fixed gesture detection for multi-touch and double-tap zooming in OsmandMapTileView .
Added version information to multicast zen ping response
Fixed bug in CompilerOptions.java by setting instrumentationTemplateFile correctly .
Replace default application mode preference with a string-based one .
Modified BoxConsumer.java to use ApiConsumerHelper.findMethod instead of ApiConsumerHelper.findMethod ( endpoint , this , log )
Change getView method return type from View to T
Added parameter &apos; numberOfSlots &apos; to constructor of DefaultMemoryManager.java
Removed unneeded InsertNode case in QueryVisitor.visit ( ) method .
Removed unnecessary &apos; useJps &apos; method in ManifestGenerationTest .
Added method &apos; body ( byte &#91; &#93; ) &apos; and &apos; body ( InputStream ) &apos; to AtmosphereRequest class .
Added flashlight tile and updated flashlight controller to check for camera availability
Updated API URL to use domain name and added API _ DOMAIN constant .
Minor change in StormBoundedExponentialBackoffRetry.java. Changed LOG.info to LOG.debug and added a missing bracket .
Change type evaluator to use ReferenceType instead of Object for better memory management .
Added import statement for Java &apos;s FileNotFoundException and NoSuchFileException classes , and updated the code to catch and handle FileNotFoundException and NoSuchFileException when deleting files during recovery .
Update preprocessWriteValueOld to use TypeInfo from getTypeInfoFromTypeString .
Added output of class alias and default cluster for console database app
Removed unnecessary imports and improved comments in EncryptedPrivateKey.java.
Update packrat dependency from 0.4.7 to 0.4.8-1 .
Fix for security vulnerability : Enforce permission check for hasEnrolledFingerprints method .
Updated AppOpsManager permissions : disallowed outgoing calls and WiFi scanning
Added data-index option to index a Wikipedia XML dump and / or a Tatoeba file .
Update ORecordIteratorCluster.java to use getStorage ( ) .getClusterLastEntryPosition instead of countClusterElements for retrieving cluster size .
Removed unused import statement for JavaElementType .
Fixed bug in LookupImpl.java where array equality was not properly checked , leading to incorrect selection of list items .
Fixed bug in HueBinding.java where brightness was not being set correctly for PercentType commands .
Modified TvView.java to replace &apos; reset ( ) &apos; with &apos; resetInternal ( ) &apos; and added a new method &apos; resetInternal ( ) &apos; to release session and reset surface view .
Added test case for non-static method with DeepLink annotation
Added copyright and license headers to IdleTimeout.java.
Changed text area painter to use preferences for tab size .
Restore cursor position when undoing insertion of text .
EditPoiDialogFragment : Changed OsmPoint.Action from CREATE / MODIFY to CREATE / MODIFY based on node ID
Added printing of URL for matches with a non-null URL in Tools.java.
Added import statement for ProjectCoreUtil and modified conditions for resolving Java and XML files .
Added support for passing arbitrary key-value pairs to the ClientApi constructor .
Added support for point lights in CubeMapMaterial
Update DeepComparator to use dataPack.getLogProviders ( ) instead of myUi.getDataPack ( ) .getLogProviders ( )
Added author Sean Owen to BookResultInfoRetriever.java
Ignoring IOException when deleting translog blob
Replace generic maps with non-generic versions for better performance .
Removed unnecessary cast in SegmentTree.query ( ) method .
Update MySQL driver to use MysqlXADataSource
Added import for StringUtil and updated name construction in PerformanceWatcher.java
Added assertion for correct value in MapStoreTest .
Added fallback value for AA API version property
Update GoRunFileConfiguration to use a shorter random suffix for generated go files
Updated Transaction class with new constant values and comments
Added import statement for IOException and updated ConnectionManager class to close the socket channel in case of an exception .
Added postWrite ( ) method call in Vec.java for proper cache management .
Removed unused imports and updated payment plugin test
Select VCS log content instead of adding it .
Disregard upper bound to suppress turn prompt also for recalculated still-opposite route
Added support for reading chunked streams till end of line .
Fix for collecting readonly files in ProjectStoreImpl
Removed unnecessary addition of dummy root to work collection in XPath.java
Added check to ensure console window is maximized before sending event
Added parameter logging in InternalIndexShard.java flush method .
Update kryo factory configuration
Added method withSelectedItem ( int selectedItemIdentifier ) to set the selected item by its identifier and updated the selection logic accordingly .
Added import statement for GroovyObjectSupport in StreamCharBuffer.java
Sort plugins by name in PluginsAdvertiserDialog
Modified GrammaticalRelation.java to change the way we print the target node in the debug statement . Instead of printing just the node label , we now print the head word node label for both the target and source nodes . This will make it easier to identify the correct nodes in the tree when debugging .
Update image width from dynamic value to fixed value
Replace &apos; this.component &apos; with &apos; component &apos; in FileDrop.java
Ensure list flags are false on failure in MovieLoader .
Added COMPACTED _ MARKER component to SSTableReader constructor .
Added icon resource ID to Action constructor in Notification.java
Added error handling for installation of Spring Boot applications .
Changed TextAttributes.merge order in printToConsole method of LanguageConsoleImpl
Changed PAGING _ TOUCH _ SLOP from 3 * TOUCH _ SLOP to 2 * TOUCH _ SLOP in ViewConfiguration .
Added max lines and set text for title TextView in ItemActivity .
Added tempFile variable for audio recording
Remove deprecated &apos; singleton &apos; property and add null check for &apos; uri &apos; property in AbstractCamelEndpointFactoryBean .
Added support for comparing dates with non-date values in OSQLFilterCondition .
Added configuration for Grails Application with null parent context
Update GLRM to use drop _ const _ cols instead of dropConsCols and add regularization to U matrix .
Added SuppressWarnings annotation to assertCurrentActivity method to avoid unchecked warning .
Replace LinkedList with ArrayList in CollectingMatchResultSink
Fixed bug in AnnotationUtil.broadcasterClass ( ) method .
Remove unnecessary reconnection service calls in CastManager .
Added favorite service for project creators with default permission template
Added support for uploading single files with custom content type and updated saveRecord method to replace $ now and $ fileName placeholders with actual values .
Added handling of EOF exceptions during writing pages to prevent logging of client disconnects .
Added NoOpDNSToSwitchMapping class to prevent DNS lookups in HdfsConfiguration .
Added import statement for AllocatedResource in CloudInstanceNotifier.java.
Removed unused import org.hibernate.validator.cfg.defs.GenericConstraintDef
Updated UidManager to use config file for uid table name
Renamed method createVisibleGraph to createVisibleGraphImpl and added a new VisibleGraphImpl class .
Added method to undemote a formerly demoted contact
RenameContactDialog.java : Changed &apos; clist.clearDisplayNameUserDefined ( metaContact ) &apos; to &apos; clist.clearUserDefinedDisplayName ( metaContact ) &apos;
Fixed error in SourceFormatter.java where { was not properly formatted
Renamed LOGTAG from &apos; TWSpannableGridView &apos; to &apos; TWSpannableGridLayoutManager &apos; .
Added test for EqualsWithDelta.java and updated matcher to compare actual and expected values accurately .
Removed unnecessary log statement in Peer.processPong
Http11NioProcessor.java : Use force = false for getPeerCertificateChain ( ) to avoid re-negotiation
Added NotNull annotation to JUnitUnusedCodeExtension and TestNGUnusedCodeExtension .
Added exception handling for deserializing ZWave nodes .
Increased LOG _ META _ DATA _ LENGTH to accommodate additional partition count .
Added try-catch block in tearDown method to handle exceptions and prevent test failures .
Added unit test mode support for renaming Python property accessor functions .
Added limit and offset limit templates for CUBRID .
Increased performance test timeout for &apos; Testing smart reformat on big file &apos; from 90 to 110 seconds .
Change CassandraArtifactCache.java &apos;s m.execute ( ) to m.executeAsync ( )
Fixed Analyzer.java to return enclosingAggregate instead of null in visitFunctionCall .
Fixed bug in NanoHTTPD.java where boundary string was not properly extracted from Content-Type header .
MavenIndex.java : Updated logging message for failed index updates
Added @ Valid annotation to RangeAggregator constructor .
Added second storage for backward compatibility in HttpConfigurable
Removed unnecessary imports and updated edge type imports in State.java.
Added check for null currentFilter before calling stopFilter ( ) and isMatching ( ) methods .
Added default value for MatchingType enum and updated StreamServiceImpl to use it .
Added &apos; getVelocity &apos; and &apos; setVelocity &apos; methods to Entity interface and deprecated &apos; getMomentum &apos; and &apos; setMomentum &apos; methods .
FavoritesTreeFragment : updated action mode menu icon
Added MAX _ ITEM _ NOSCROLL _ COUNT constant and updated LIST and LIST _ ITEM static final variables in UIDropdownScroll.java
DecayingDelayTripTimes : Changed constructor parameter from &apos; true &apos; to &apos; false &apos; for linear decay
Replace List with Set for getRelevantDependencyArtifacts method
Renamed method from &apos; getWhereClause &apos; to &apos; getWhereCond &apos; in class Select
Added support for displaying HTML content in BookInfoActivity .
Added @ JsonIgnoreProperties annotation to Job class
Added logging for failed page fetches due to large page size or unknown errors .
Added closer.close ( ) to ensure all open files are closed before renaming jar file
Added test case for computing silhouette coefficient .
Added synchronized block for mRulesLock and added Binder.clearCallingIdentity ( ) and Binder.restoreCallingIdentity ( ) for setting uidPolicyUncheckedLocked .
Added isApnTypeActive method to GsmDataConnectionTracker and fixed getActiveApnTypes method to return all active apn types .
Added check for classes.dex file in decodeSourcesSmali method
Added finish ( ) method to TermSession and TranscriptScreen to release resources properly .
Added support for inline parameters with CharSequence values .
Added support for parsing boolean values in DynamicProperty .
Added &apos; seen &apos; and &apos; unwatched &apos; actions to TraktTask , and updated the watchlist action to use the new &apos; MovieService &apos; class .
Fixed issue with OperatorMerge draining queue completely when requested &gt; 0 .
ToolbarActionBar.java : Always return true for onKeyShortcut to block temporary panel
Added updateExpandButtons method call in NotificationContentView to update expand buttons when the notification is expanded or collapsed .
Replace LocalFileSystem with StandardFileSystems
Replace if / else with switch case in MultiMapOperationFactory
Added a test for 302 redirect with followRedirects set to true
Added onDestroyView and setItemAlignment methods to RowsFragment and RowsSupportFragment to properly clean up views when fragment is destroyed
Update site domain string in DeleteSiteDialogFragment
Fixed bug in HistoryGuru.java and IndexDatabase.java
Removed unnecessary &apos; protocol.sendShutdown ( ) &apos; call in OClientConnection.close ( ) method .
Modified SuppressParameterFix to use psiParameter.getTypeElement ( ) instead of JavaSuppressionUtil.canHave15Suppressions ( psiParameter )
Added ReaderPostDetailFragment to fragment container and added it to the backstack .
Modified RequestServer.java to make registerRequest method public .
Update ThemePreviewFragment to use desktop user-agent for preview URL
Added resource leak detection and closing in AbstractDnsMessage # deallocate ( )
Added backpressure support to Observable.window ( ) method .
Fixed OreDictionary.getOres ( ) method to use getOreID ( ) instead of nameToId.get ( ) for performance reasons .
IdeFinder.java : Changed Windows detection to match &apos; windows &apos; instead of &apos; windoz &apos; .
Added import statement for cli.System.IO.Directory ; and updated localPath to use Directory.GetParent instead of Environment.SpecialFolder.MyDocuments
Replace System.err.println with Debug.e in SocketUtils.java.
Added null check for unblockedRating in TvInputManager # requestUnblockContent
Added new import statements for WindowManager.LayoutParams and updated sublayer values for application media and overlay .
Modified ExplainRequestBuilder to include fetch source and operation threaded options .
Added generic type parameters to DrawerBuilder and MiniDrawer classes .
Removed intentional failure in TestRequest test case .
Added newline character to notification text for running tests
Added check for cursor.moveToFirst ( ) before getting image data
Added JSON parsing for magic link sign in error responses
Removed deprecated jspError methods and updated Dispatcher.java
Removed unnecessary import of AccelerateDecelerateInterpolator
Removed unnecessary &apos; return &apos; statement in SwiftInputStream.java.
Increased timeout for nested creation test from 300ms to 3000ms to account for slow CI environment .
Added support for web-mode in GwtPlatform by replacing Array with Java arrays and added a new method to resize arrays .
Added unsubscribe method to AbstractViewModel using SubscriptionAnnotationUtils .
MavenModuleImporter : removed unnecessary tries-with-resources statement
Added code to replace inline tags from Wikipedia Atom feed items .
Update build number to SNAPSHOT
Removed unconfirmed removal of tokens for leaving endpoints
Added GrowlNotificationActivator.java file and implemented Growl Notification service .
Fixed bug in HttpResponseChannel.java where buffer was not being freed after flush .
Removed unnecessary onWindowFocusChanged method and updated UnlockPatternListener to reset the pattern when the timeout dialog closes .
RectangularShape : Changed check for setting base size to consider both width and height .
Remove unnecessary code in DTDModelLoader.java
Throw IllegalArgumentException for invalid lower / upper bounds in GLM
Make lastUpdateTimestamp volatile for faster expiration task processing
Update duplicate file error message format
Renamed getUserId ( ) to getUserAuthId ( ) and added a comment explaining the purpose of the method .
Fixed binding of LifeLine server socket to local host .
Update MemoryCircuitBreakerTests to use greaterThanOrEqualTo instead of equalTo for breaker.getUsed ( ) comparison .
Added a validation for non-blocking writes with the NIO connector on Linux , and increased the pause time for write operations from 7 to 30 seconds to account for slower performance .
Removed unnecessary log statements and improved error handling in ForecastFragment .
Modified MidblockMatchState class to change the logic for choosing the next state when reaching the end of an edge .
Remove unused &apos; cassandra.stop ( ) &apos; call in DeleteTest.java
APNS.java : Made class final
Added nullability annotations to MongoStats class
Remove unnecessary cast in ArrayIterable.java
Added Netty-ChannelAlive-Timer to ensure channel alive
Minor change in GenericFileConsumer.java : removed &apos; pre &apos; from &apos; prePollCheck &apos; and changed &apos; pool &apos; to &apos; poll &apos; in two places .
Fixed missing umbrella element path validation for ConfigElement .
Increased maximum row size in SelectionOperatorUtils.reduce ( ) from 1000 to maxRowSize
Added getRotation method to Sprite class and modified setRotation method to include rotation relative to origin .
Removed unused rotation tab and added color tab to options dialog .
Removed unnecessary TODO comments and cleaned up MapInterfaceTest.java.
Modified WorldEditListener to use a simpler syntax for the &apos; / / smooth &apos; command and updated the corresponding help message .
Fixed issue with too much text in ConsoleViewImpl
Remove unnecessary constant definitions in RILConstants.java
Changed interface from ExportedDialog to ApplicationWindow for ConfigurationManager and PopupDialog
InternalEngine : added checks for version map size and gc deletes in onSettingsChanged ( ) , and removed unnecessary refresh in IndexShard .
Fix for Utils.delete ( ) method to properly delete files on Windows .
Calculate output hash and size for locally built rules .
Modified AbstractContextSelectToolbarStatusPanel.java to change protected abstract Component switchViewForContext ( Context context ) to protected abstract void switchViewForContext ( Context context )
Added sorting of countries by views in GeoviewsModel .
Fixed issue with activity leaking window android.widget.ZoomButtonsController $ Container by setting visibility of WebView to GONE in onDestroy ( ) method .
Remove unnecessary receiver registration in MainService .
Added isArray method to ClassReflection and updated isStaticClass to use Modifier.isStatic instead of ReflectionCache.getType ( c ) .isStatic ( ) for better performance .
Added getLegacyVpnInfoPrivileged method to Vpn class .
Added check for uninitialized via count in ViaInstruction.getViaCount ( )
Fixed null pointer exception in View.setLayoutParams ( ViewGroup.LayoutParams params ) by checking if params is null before calling requestLayout ( )
Modified ServerControllerService.java to use subTarget instead of container for TrackingServiceTarget creation .
Fixed CollectionUtils.join method to handle null or empty collections .
Updated OGlobalConfiguration to change the interval for dumping profiler values from seconds to regular intervals .
Added printSupertype field and setter method to BeanSerializer .
Added country code field and getter / setter methods to FakeAddress class .
Fixed issue with SQLForeignKeyManager.java - Now using DBUtils.getQuotedIdentifier ( attribute ) instead of attribute.getName ( ) for correct identifier quoting .
Added isValidAfterUnsafeCheck method to FqName and made isSafe ( ) method return true if the name is safe .
Removed unnecessary callback handling in ResponseNotifier .
Removed unused import : org.apache.derby.client.am.SqlException
Remove unused searchOrder field and simplify platform container construction .
Added @ Ignore annotation to ReferrerPushStrategyTest
ImapStore : Remove unnecessary &apos; false &apos; variable and simplify code .
Reduced maxWaitQueueSize for better performance .
Increased maxMessageSize to 16MB in OkHttpChannelBuilder and TestServiceServer
Added import statement for OGlobalConfiguration and increased NETWORK _ LOCK _ TIMEOUT to 15000
Fixed issue with incorrect file indexing by calling ProjectCoreUtil.isProjectOrWorkspaceFile ( ) instead of manually checking file type and directory structure .
Remove unused method isFilesWordPressCom and update isWordPressCom to take a URL parameter
Added @ ClassRule for disabling HDFS cache in JournalShutdownIntegrationTest
Removed unnecessary imports : ArrayList and List
Remove unused variable &apos; intermediate &apos; from CachingClusteredClient .
Update LocalNotification.java to use apply ( ) instead of commit ( )
Added logging for DeleteReturnRoute command failure .
Fixed date conversion test to use Locale.US
Added check for XHTML + XML MIME type in LoadListener
Added ability to add misspelled words to user dictionary
Added additional registration for chat handler with service name replacement
Added import statement for DanmakuFactory in DanmakuFilters.java
Update resource locations for static resources and add trailing slashes to resource paths .
Added try-catch block to handle ClassCastException in FeedEntryFilteringService # filter method .
Removed unused import and abstract method , and fixed typo in FilteredQueueAdderListener.java
Added import statement for org.neo4j.io.fs.FileUtils and added check for Windows OS in backgroundThreadsMustGracefullyShutDown test .
Update Sketch.java to reset all state information for the sketch object and rename recent files .
Added method toString ( ) to InMemoryLookupCache class and modified incrementWordCount ( word ) method to incremente word count by 1 .
Update ChangeParameterMetaDataTest to correct database support for replaceIfExists parameter .
Fix focus search issues in DirectoryFragment and FocusManager
Update license header for IOUtils.java to reflect Apache Software Foundation ownership .
Remove serialVersionUID field from WorldEditPlugin.java
Added imports for Spring Web and HTTP annotations in SpringMvcCompilerAutoConfiguration .
Added ability to expand or contract an item on right-click , and updated selection handling .
Removed unused variable &apos; range &apos; and removed unnecessary code in LayeredLexerEditorHighlighter .
Added support for setting plus icon color through @ ColorRes id .
Added name parameter to getSdk method and updated mockJdk and fullJdk methods to use it .
Added support for Groovy getters and setters in GrClassImplUtil .
Added constant FRAME _ TIME _ MILLIS and updated calculation of frameIndex
Added support for setting attributes in MemberAttributeConfig using putAll ( ) method .
Fixed issue with incorrect clipping in ViewPositionAnimator .
Added support for basic authentication with Apache Tomcat .
Fixed bug in SuperColumnSerializer where localDeleteTime was not properly validated , leading to an IOException .
Added import statement for android.text.TextPaint and updated IconicsDrawable.java to use TextPaint instead of Paint for icon painting .
Added checks for null or empty username and password before login ( ) and added throwNotConnectedExceptionIfAppropriate ( ) and throwAlreadyLoggedInExceptionIfAppropriate ( ) to AbstractXMPPConnection .
Fixed logging statement in TwoWayView.java
Added check for speaking route in BinaryRoutePlanner .
Replace &apos; mNotesAdapter &apos; with &apos; getNotesAdapter ( ) &apos; in NotificationsListFragment .
Replace String with CharSequence in TextChange .
Replace MyChildViewHolder with MyChildItem in onBindChildViewHolder
Removed unnecessary AnalyticsTracker.track call in ReaderPostPagerActivity # onPageSelected
Removed unnecessary import and fixed column name in InheritanceManyToManyForeignKeyTest .
Added lazy loading of sessions using ExchangeSessionManager
Update SlidingPaneLayout.java to modify slide range calculation and add parallax effect .
Added DetachVolumesChaosType to enabledChaosTypes in BasicChaosMonkey
Added handling for ContainerNotFoundException when stopping containers .
Added tests for invalid method injection in ActivityWithInvalidMethodInjection .
Modified MessageView.java to change the logic for determining whether a message is downloading from partial to full .
Added nullability annotations for LexerATNConfig.lexerActionExecutor
Added method isBatchUseTransaction ( ) to SqliteAndroidDatabaseType , and set it to return true .
Added a stub for Java &apos;s built-in disjoint ( ) method to help with discovery .
Disable flaky peer priority test
Added methods isDeferred ( ) and isDynamic ( ) to TldAttributeDescriptor interface .
Added method getJaasAuthenticationCallbackHandlers ( ) to JaasAuthenticationProvider class .
Fix for DexFileFactory.loadDexFile ( file , api ) ;
Removed unnecessary &apos; and &apos; method in RedirectConfig
Added installerUid to VerificationParams constructor
Added one more value to the x-axis of the scatter chart .
Update StatsService to use a single thread pool executor for updating UI to prevent sluggishness .
Added QUIT command execution in SmtpTransport to improve shutdown behavior .
Removed unnecessary files from SourceFormatter .
Hide &apos; unread items &apos; group if empty
Simplified serving of resources in DefaultServlet by removing unnecessary try-catch block and handling of broken pipe exceptions .
Added tracing for indices.recovery and action.search.type in GeoBoundsTests .
Added @ SuppressWarnings annotation to LocationTest.java
Fixed issue with caching local files content in PersistentFS
Replace &apos; int &apos; with &apos; Integer &apos; in Transition.removeTarget ( int targetId )
Fix for # 12345 : Reverse parent and child order in DependencyGraphBuilder # addParentSpecificArtifacts
Modified FileSystemBlockLocationIntegrationTest.java to correct the offset and length calculations for testing block locations .
Added bulk selection feature to comments list fragment .
Added nullability annotation to visitChildrenRecursively method return type .
Fixed bug in Call.java : removed unnecessary code for Ruby &apos;s Class.new
Replace LinkedHashMap with Map in DatastoreImpl.merge method .
Deprecated getOOoLabel method in Tools class .
Add retry logic for requestSpace method in RetryHandlingBlockWorkerClient
Added ignore path of root directories in TaskFilePropertyPathSensitivity .
Request focus on contact list panel when selecting a protocol
Added check for ProtocolProviderFactory.FORCE _ PROXY _ BYPASS property to determine if proxy check is disabled and skip registrar connection check for sipXecs , sipXcom , ...
Added logging statement for wallet balance and receive address before testing fee level .
Fix bug in IndexedSet.java : change &apos; index &apos; to &apos; indices &apos; in exception message
Flipped the order of parameters in isFieldKept ( ) method for OptimizerNode.java
Removed unused variable &apos; index &apos; in TypePool interface
Removed unnecessary System.out.println statements in AbstractSecurityInterceptor .
Removed unnecessary import statement in TestUEncoder.java
Update format of FILE _ IN _ SRCS _ AND _ HDRS _ ERROR _ FORMAT from &apos; File % s is in both srcs in hdrs &apos; to &apos; File % s is in both srcs and hdrs &apos; .
Added null check for GraphDatabaseShellServer before shutting down .
Update foldLeft and foldRight tests to use &apos; ! &apos; as separator .
Removed unnecessary operation execution in ParseAndMarshalModelsTestCase .
Updated RemoteImageLoader to post loaded images to TextView instead of ImageView .
Added ReflectiveConfigOptionLoader.loadOnce ( ) to TitanFactory &apos;s Builder constructor .
Added check for unset parameters in ImageTagFactory and added test case for it .
Fix lifecycle management in Node class .
Removed unnecessary UserManager.hasUserRestriction ( String ) check in Settings.java
Added currentSheetViewHeight variable and updated OnLayoutChangeListener to track changes in sheet view height .
Fixed bug in setSelectedTabView ( ) where it would crash if there was no selected tab .
Rename Register.java to RegisterResourceRoots.java and update import statements
Remove Czech language from ignored languages in PatternRuleTest .
Fix null pointer exception in FileBackupHelper
Updated timeout values and screenshot save path in Config class .
Added a new line break before the LayoutLocalServiceUtil.addLayout call to improve code readability .
Added method isBikeParkAndRide ( ) to OSMNode and OSMWithTags classes .
Added a note to clarify the behavior of WifiInfo.getSSID ( ) in different API levels .
Added null check for array comparison in ProtocolToModelAdapter .
Updated NetTest.java to add new randomized variables and updated values for existing variables .
Added handling for unsupported scheme errors in CordovaWebViewClient .
Added Java language implementation and isJavaFile method
Splitter EIP : Changed response from latest message returned from destination endpoint to original input message with custom AggregationStrategy .
Create job manager in JobBootReceiver
Replace requestIntent with the last element of intents array
Added estimation of time left for training in NNModel .
Added support for Hibernate Search in QueryDSL by creating a new SearchQuery class .
Removed unnecessary fields from VcsLogRefresherTest .
Fixed security issue in ActivityStackSupervisor by checking for FLAG _ ALLOW _ EMBEDDED flag before embedding activity .
Removed unnecessary removeAll ( ) call in CodeFoldingConfigurable .
Added check for current printer before setting destination spinner selection
Simplify detection of cycles in SkyframeBuildView
Added visibility checks for download progress and duration in NewEpisodesListAdapter , and fixed FeedItem to use media.hasEmbeddedPicture ( ) instead of just media .
Added static final modifier to IncludeInfo.EMPTY _ ARRAY
Added support for JavaCoverageEngine in DefaultJavaCoverageRunner
ExtensionAlert : Added check for null view before updating site tree .
Added debug statement for Hadoop version checking in BaseHadoopTest .
Added check for first edge in GraphPath to determine last Edge was Street
Added check for extended class name matching in MutableExceptionCheck .
Fix illegal argument exception message for missing processor type .
Update DozeService.java to reregister for pulse notifications after requesting a pulse .
Added daemon flag to ThreadCpuStats collector thread
Added web references for input : files in PApplet.java
Update ConfigurationLoader.java to use getClass ( ) .getClassLoader ( ) instead of Thread.currentThread ( ) .getContextClassLoader ( ) to load the internal dtd resource .
Added ForOwnerType annotation reader for reading an owner type .
Added temperature scale conversion for ZWaveMultiLevelSensorConverter .
Added JpaTraceEventMessage import in JpaTraceEventMessageTest .
Added support for cursor window in SQLiteConnection.nativeExecuteForCursorWindow.
Added import statement for FunctionInvoker and modified castToVarchar method to use it .
Added check for isFinishing ( ) before showing toast or auth alert in PostsActivity .
ConfigImportHelper.java : Added check for main jar name in isInstallationHome method .
Modified MetaObjectProtocol.java to change parameter type of getMetaMethod method from array of argument types to array of argument values or corresponding argument types .
Added support for deferred re-painting of children in PaintingParent interface and added Wrapper class to implement PaintingParent .
Added filename suffix to MimeUtility.java
Update ScanFilterAndProjectOperator to use getCompletedBytes ( ) instead of incrementing completedBytes
Added public access modifier to getRootModel method in ModuleRootManagerImpl .
Added fireWriteComplete ( ) call in LocalChannel &apos;s handleUpstreamMessage ( ) method .
Bitmap _ Delegate.java : Changed create Bitmap with nativeInt and isMutable parameters .
Increased threshold for BalancedShardsAllocator from 1.1f to 100.0f in RecoveryBackwardsCompatibilityTests .
Added intersection check for dirty regions in HardwareRenderer .
Fixed ExcelImporter.java to correctly handle rows with empty cells .
Fixed typo in GsmDataConnectionTracker.java
Added methods for displaying toast messages on UI thread .
Update getTableName ( ) method to use config instead of hardcoded table name
Added method outputBaseTime ( long ) to log base time in seconds since epoch .
Renamed SimpleClass to SimpleClassCustomName in CustomNamedFactory and updated create ( ) method return type .
Throw BlockStoreException when setting chain head
Fixed DNS lookup for ServerDialback connections
Added method isGoModule to GoSmallIDEsSdkService to check if a module is a Go module and its SDK home path is not null .
Removed unused import android.support.v4.view.ActionMode.Callback ; in ActionBarCustom.java
Added import statement for org.springframework.core.io.ClassPathResource in MockApplicationContext.java.
Added transaction handling for proxy methods
Sketch.java : updated editor.setCode ( ) method to use System.currentTimeMillis ( ) for visited flag .
Added hasHighlightedFields ( ) and getHighlightedField ( ) methods to MessageResult class .
Added support for deprecated fields in JavaPresentationUtil
Remove unused import and adjust thread priority .
Added check for null magnification spec before printing it in DisplayContent.java
Added hostname equality check for GangliaReporter spoofing .
Update BeanPath casts to use ConcurrentHashMap for better performance .
Added method toString ( ) to ToroDocument class .
Added generics to ContainerUtil.addAll and added a new method set .
Increased sleep time after vertex visit from 10 to 50 and removed unnecessary variable
Updated code style settings for committed documents .
Added rolling count for fallback missing events in JsonMappers
Added method closeTransactionConnection ( PreparedStatement , Connection , boolean ) to close a PreparedStatement and Connection , and added a new method closeTransactionConnection ( Connection , boolean ) to close a Connection , both with transaction rollback or commit depending on the value of abortTransaction .
Removed unused variable ByteBufferUtil.TOTAL _ MAPPED _ BYTES and related logs .
Added ReplicaSet category to DBTest and updated test method to use ReadPreference.secondary ( ) for executing command with secondary read preference .
Added face detection for full-size images
Fixed issue with zero-width history in LanguageConsoleImpl .
Added import statement for &apos; com.intellij.openapi.util.Pair &apos; and changed return type of getTextRange method to Pair &lt; PsiElement , TextRange &gt; in RegExpRangeProvider interface .
Added import statement for KotlinFrameworkDetector and updated isConfigured method to check for Java Kotlin module
Added fireEvent method call to DBNDataSource.setName to notify listeners about update
Added RRC to the list of parents for PP nodes in EnglishGrammaticalRelations.java.
Added onTouchListener to MainMenuPopover to dismiss when tablet-sized device
Added check for empty personal string before constructing address string .
Enable zooming and pinch gestures for SSO webview dialog .
Modified DeepWaterFrameIterator to use adapted frame key for cache marker
Added execute ( ) method to IgnitedAsyncTask that takes no arguments and suppresses warnings for nullable types .
Added method isDestroyed ( ) and print statement for FragmentManagerImpl .
Fixed bug in AfcClientTest : added missing close ( ) call for RandomAccessFile and updated calculation of write speed .
Add trace logging for WebSocket.write ( ) and handle binary writes with transformations
Added progress indicator for directory scanning in FileBasedIndexImpl .
Added null check for filename parameter in ExifInterface constructor
Fixed issue with null headers in GenericMessage constructor
Fixed SSLContext initialization to handle null key store and trust store locations .
Added FeatureMethodEarlyBeanProxyTests to FeatureMethodLifecycleIssueTestSuite
Modified RequestLoggingFilter to use requestSpec.getMethod ( ) .toString ( ) instead of ctx.getRequestMethod ( ) .toString ( )
Fixed issue with incorrect semi-colon usage in Groovy code
Renamed &apos; Retrieving filtered results from the VCS &apos; to &apos; Looking for more results &apos; in VcsLogDataHolder.java
Rename parameter &apos; project &apos; to &apos; project &apos; and add &apos; NotNull &apos; annotation
Added exception handling for orphan blocks in WorkerStorage
TextureView : updated layer update logic and added transform matrix application
Added &apos; throws SuspendExecution &apos; clause to method tryGetActor ( String name ) in LocalActorRegistry class .
Removed unnecessary call to ReflectionUtil.resetThreadLocals ( ) in ApplicationImpl .
Updated AvsAnRule.java to load exceptions from external files det _ a.txt and det _ an.txt.
Removed unnecessary highlighting of collecting data message in XValueContainerNode .
IncArtifactBuilder.java : Converted file paths to system-independent names .
Changed catch block from IOException to Throwable in FileProcessingCompilerAdapterTask .
Fix minor typos in Version.java and FinderPattern.java
Update DatePart function to expect a date and a string instead of a number
Removed unnecessary setUserData ( null ) call in destroyFixture method .
Added convenience method to register proto compile action and fixed registration process .
Fixed bug in ClassDefinition.java where &lt; clinit &gt; method was not properly handled
Removed unnecessary logging statement in HttpRequestBuilder .
Fixed issue with node coordinates having negative values .
Added logging statement for unable to read rackdc property file
Added timestamp comparison with zone offset check in Result.java.
Modified WorkspaceFileFunction to use FileValue instead of null checks for workspace files .
Added Timber logging for custom events and errors
Added method implementations for writing byte arrays and arrays of bytes to CountingOutputStream .
Added check for peerCount in PeerEventListener methods
Added import statement for android.annotation.SuppressLint and updated WPDrawerActivity.java file to use SuppressLint annotation for targeting API level 21 and above .
RestClientUtils : Send login email endpoint updated from &apos; send-login-email &apos; to &apos; auth / send-login-email&apos;
Modified GeneralToSMTRunnerEventsConvertor.java to change onTestFailed to onTestIgnored in fireOnTestIgnored method .
Fix for issue # 12345 : Removed unnecessary &apos; given &apos; statement in ConstructorInjectionTest .
Added Selection End Offset to CompletionUtil &apos;s emulateInsertion method
Added recovery logic for NettyAsyncHttpProvider
Update MacHashFunction.java to use prototype.getMacLength ( ) instead of getMac ( algorithmName , key ) .getMacLength ( )
Increased timeout for JDBCExtractionNamespaceTest from 500 to 60 _ 000
Moved temporary directories from / tmp to . / target / tmp
Added debug logging for AfterInvocationProviderManager
Removed synchronized block in StringManager.getManager ( String packageName ) and made gmtZone static final .
Added buffering start and end events to MediaPlayer
Fixed issue with incorrect type usage in DropTypeStatement
Added @ Ignore annotation to BatchEndpointIT.requestsShouldNotFailWhenHttpLoggingIsOn ( ) test method .
Added awakenScrollBars ( ) method call in onMeasure ( ) to enable overscrolling
Added imports for UI widgets and updated CreateGameScreen class .
Added null check for summary.notification in NotificationGroupManager.toString ( ) method .
WebView : updated loadDataWithBaseURL method to use baseUrl for resolving relative URLs and applying JavaScript &apos;s same origin policy . Also added a note about content accessing local device files .
Added method setSystemUiVisibility ( int visibility ) and getSystemUiVisibility ( ) to return the status bar visibility that this view has requested . Also added a listener to receive callbacks when the visibility of the system bar changes .
Fix mod sorting exception handling in Loader.java
Fixing equality check for Versioned class
Added SuppressWarnings annotation to GcStats.minorCount field
Fix for child view performClick not being called in MaterialRippleLayout
Added ability to create relationships with variable length in ShellTest .
Fixed exception handling in LaterInvocator to wrap everything in a RuntimeException for better stacktrace and to support TC ComparisonFailure feature .
Added support for xts objects in RObjectEntry
Added debug logging for data and value in BindyCsvFactory
Added atomic long field updater for request count in BackpressureUtils .
Added import statements for org.parboiled.matchers. * and removed unnecessary imports .
Idempotently cancel pending timer in SenderChannel and TimerWheel .
Added max zoom warning for Edit Tilesource feature
Material : updated parameter application to use &apos; apply &apos; method instead of manual texture and uniform updates .
. CherryPicker : added static final String CHERRY _ PICK _ HEAD _ FILE = &apos; CHERRY _ PICK _ HEAD &apos; ; removed unused variable ; updated import statement . ( cherry-pick # 1234567890 ) &#91; ci skip &#93;
Removed unnecessary check for empty line in FileTypeUtil .
Added debug logging for requests that yield responses
Added check for hasDrawnLw ( ) in WindowState.java to ensure that a window is not removed before it has been drawn
Added log statement for public notification in BaseStatusBar
Update path info from / a to / z in CustomAnnotationTest .
Removed unused code for Blaze modules that allow some action inputs to be missing .
Removed unnecessary import of Uri in Sticker.java
Added warning for &apos; this &apos; reference in arrow function inside object literal
Fixed DhcpClient to properly handle lease renewal and rebinding .
Added check for length of 1 in StringToCharacterConverter
Added a check for tied methods in getMatchingParams
Added unique constraint on book _ id , type , and uid columns in SQLiteBooksDatabase
Added test for cache local entries with Near Cache
Added support for parsing disable modes in KeyboardShortcut constructor .
Remove test for @ JsonIgnore with @ ApiModelProperty
Added updateMediaLocalToRemoteId method to WordPressDB and updated MediaUploadService to use it instead of deleteMediaFile .
Remove connection from dispatcher when it becomes inactive .
Modified SessionFixationProtectionStrategy to only create a new session if the client &apos;s requested session Id is valid .
Update Picasso.java to use more accurate language in method descriptions .
Added new fields to CallLog and ContactsContract classes .
Added @ Category ( SlowTest.class ) annotation to LockTest.testLockConditionSignalAllShutDownKeyOwner and updated HazelcastTest to include @ Category ( QuickTest.class ) annotation .
Added space after type parameters in PrettyCommentsPrinter and fixed type parameter handling in ValWeirdTypes .
Updated AbstractIndexCountBasedRetentionStrategy to use get ( ) method for retrieving maximum index limit .
Added AjpIgnore annotation to PreChunkedResponseTransferCodingTestCase
Added mStructureChanged and markKnownViewsInvalid ( ) calls to RecyclerView.java to handle structure changes .
Added writing of test name to media memory output file in MediaPlayerPerformance test .
Increased default value of mainThreads to match Netty &apos;s default .
Change histogram interval from hour to minute
Fixed finally block in DropwizardTestSupport.after ( ) method .
Replace YearMonthDay with YearMonthDay in documentation
Increased search word range by 1 character .
Added GearLoadingRenderer to CircleRotateActivity .
Added support for top sleeping state in ensureForegroundActivity
Renamed &apos; Clicker &apos; class to &apos; Clicks &apos; and updated method names to use Java 8 syntax .
Added check for SVG header in isSVGImage method
Added test cases for executing Gremlin scripts with line breaks and null values
Replace TableObserver with createTableObserver in AppearanceUIController
commit - Fix for issue # 12345 : Remove unnecessary code in ActivityStarter.java
Fix for dropping stash after resolving conflicts .
Update MaterialDialog.java to use ContextThemeWrapper for inflating custom views .
Updated JobCreateCommand to use a single path for data volumes and added support for read-write and read-only volumes .
Renamed Builder constructor to private .
Rebuild all the inheritance and set all the linked classes for OConsoleDatabaseImport .
Fixed escape sequence handling in AstLiteralExpression .
Added import statement for DefaultProducerServicePool in XmppRouteTest.java
Added methods for translucent navigation and tint forcing .
Added @ Ignore annotation to TopicOverloadAbstractTest # whenBlock _ whenNoSpace test method due to issue # 6819 .
Removed unnecessary System.out.println statement in ManagementCommunicationService.start method .
Updated JavaPredefinedConfigurations to use &apos; Variable + &apos; instead of &apos; Variable * &apos; for static fields without final modifier .
Added check for contact &apos;s roster status before creating a conversation
Added import for &apos; R &apos; class in FastAdapterUIUtils.java
Added import statements for Instant and DateTimeFormat , and updated the illegal argument exception message to include the instant value in the format &apos; yyyy-MM-dd &apos;T&apos;HH : mm : ss.SSS &apos; .
Fixed non-deterministic test for max outer concurrency .
Added import statement for &apos; com.intellij.util.containers.OrderedSet &apos; in AndroidRootUtil.java.
Removed unused imports : org.zaproxy.zap.extension.pscan.PassiveScanner and org.zaproxy.zap.utils.HttpUserAgent
Added check for EmptyIntentionAction in CleanupInspectionIntention.isAvailable
Added notification mechanism for caching blocks in worker
Added error message to illegal state exception in install method of ByteBuddyAgent
Added requireNoContent method call in RelativeOrderingMetaDataParser and removed unnecessary WarStructureDeploymentProcessor and WarAnnotationIndexProcessor in WebDeploymentActivator .
Fixed error message in StudyUtils.java from LOG.error to LOG.info
Removed unnecessary imports and code cleanup in UIDialogCreateNewWorld.java
Added code to handle rare case of smb joins in Utilities.java
Removed unused &apos; setAcquireIncrement &apos; method call in CreationTest .
Remove unused variable &apos; i &apos; and update type imports
Added border creation for JBScrollPane
Fixed issue with StripesHandlerPlaceBlock.java - now properly checks onItemUse instead of placeBlockAt .
Added import statement for ModelTestUtils and updated validation of model descriptions .
Increased timeout from 5000 to 10000 in Parser.java
Added shortcut for applying left or right side changes in TwosideBinaryDiffViewer .
Removed unnecessary import statement for &apos; JsonDeserializer &apos; in &apos; TextContainerDeserializerHelper &apos; . &apos; ,
Removed unnecessary System.err print statements in BugPatternFileGeneratorTest .
Fixed printObservableOnCompleted and printObservableOnUnsubscribe to use anyString instead of Thread.currentThread ( ) .getName ( )
Update Editor.java to mark text display lists as dirty when updated
Removed unnecessary &apos; value &apos; method from Mongo interface .
Fix bug in CreateAliasExecution : check if remote table exists before creating alias
Update Cassandra helper to use client instead of entity
Refined callback interface and cleaned up login / logout interface .
Fixed issue with incorrect restore offset in StaggeredGridView .
Fixed formatting issue in PhoneStatusBar.java.
Added @ SuppressWarnings ( &quot; PMD.PreserveStackTrace &quot; ) to JMXConfiguration.setConfiguration method to avoid PMD warning .
Added show ( ) method to MaterialDialog class .
Renamed SimpleGraphServiceImpl to GraphServiceBeanImpl and added a zero-arg constructor .
Fix for Windows &apos; start &apos; command behavior of dropping anchors from the URL for local URLs .
Added parameter &apos; size &apos; to auxAllocateAndAdjustFields method .
Added check for non-default sequence number in Transaction.setLockTime ( ) method .
Added main method to CommandLineOptions class for printing manual page .
Updated LOOP _ INC _ VALUE from non-final static variable to final constant field .
Update TaskManagerConfig to increase infoUpdateInterval from 200 milliseconds to 1 second .
Fixed equality check for AccessibilityWindowInfo.mBoundsInScreen
Added support for indeterminate progress bar in AccountActivity .
UI _ Controller.java : updated over method to use &apos; prevGraphElement &apos; variable for optimization
Added positive response field and updated serve method in CreateFrame.java and FrameCreator.java
Added layout direction to PhoneWindow &apos;s decor and title view
FloatingDecorator : added support for alpha mode
Update Build.java to use SystemProperties instead of hardcoded values for supported ABIs .
Replace HashMap with LinkedHashMap in OFunction class .
Added parameter NanoHTTPD.IHTTPSession to serveFile method in MarkdownWebServerPlugin .
Added settings for Kuromoji plugin and updated nodeSettings method to include plugin settings .
Added support for FOCUSABLES _ TOUCH _ MODE in View class .
Added visibility check for Gitignore model in NewRepositoryPage
Added constructor with throwable cause to CancelTaskException and ProducerFailedException .
Update AnalyticsTrackerMixpanel to use capitalized event names .
Added support for day and night modes in RulerControl .
Change catch block from Exception to Throwable in Loader.java
Update FileSystemMasterIntegrationTest to use AuthorizedClientUser instead of hardcoded username for tests .
Update ExternalToolPassFactory to use FileStatusMap.getDirtyTextRange instead of file.getTextRange ( ) for external tools highlighting .
Update DefaultSubscriptionBase to use chargedThroughDate if it exists and is after current UTC time , otherwise use current UTC time .
Added selection background color to TaskExecutionTreeTable
Fixed bug in DataInfo.java : _ numOffsets was not being correctly initialized when _ nums was 0 .
ElasticsearchIntegrationTest.java : Remove unused &apos; GLOBAL &apos; scope and update exception message for unknown scope .
Throw OAuthException for UnknownHostException instead of setting code and body
Removed stack trace from error messages for retrofit and TheTVDB errors
Fixed logging statement for push notification
Updated MachineReading.java to use MachineReadingProperties for percentageOfTrain and added missing newline at end of file .
Added method &apos; useGraphConfig &apos; to BulkLoaderVertexProgram to allow configuration of graph settings .
Added import statement for InvalidRecipeException and updated catch block to handle it .
Added check for null buffer in getAllocationByteCount ( ) to properly handle native-backed bitmaps .
Update MongoDBStorageProvider.java , set default value for &apos; name &apos; field .
Added sleep ( ) method call to Clicker.java to improve scrolling performance
EditSession.java : Changed generation coordinates for big trees from ( x , y + 1 , z ) to ( x , y , z )
Added custom variable &apos; build &apos; to ConfigApi.FetchConfigRequest and updated callback to include promo information .
Added implementation of toString ( ) method in CarrierIdentifier class .
Removed method eventCodes and updated eventBufferSize method .
Obsolete WebIconDatabase and WebSettings interfaces
Added public access modifier to IntersectionOperator and UnionOperator classes , and made PartitionFilter class abstract .
Added &apos; isMovedTemporarily &apos; method to StatusResultMatchers
Added check for MAX _ INACTIVE attribute in IdleResourceInterceptor
Added import statement for ProgressManager
SettingActivity : Removed EditTextPreference for notification ringtone and replaced it with a Preference
Fixed SSL handshake issue by removing channel dependency
Added method getTraceStream ( ) and setTraceStream ( OutputStream ) to LexerATNSimulator class .
Added method replaceUnicodeDigits to normalize phone numbers with arabic / unicode digits .
Added method setUsePooledMemory and setNettyTimer to DefaultAsyncHttpClientConfig
Added PIN support for OAuth updates .
Removed unused variable &apos; myFoundComponent &apos; in KeyProcessorContext .
Removed unnecessary imports and corrected variable names .
Removed unnecessary import statement for &apos; java.util.HashMap &apos; in TokenEndpoint.java
Added exception handling for cache creation and saving in UncachedSpiceService .
Added import statement for com.intellij.util.containers.ContainerUtil and updated code to use ContainerUtil.addIfNotNull instead of adding null values to openPaths list .
Added CDMA roaming mode configuration
Fixed issue with closing project in LightIdeaTestCase .
Fix unused imports in InnersOnDemand1.java and InnersOnDemand2.java
Added media type checks for XML and JSON in DiscoveryJerseyProvider
Increased performance test timeout for detecting indent on hot file from 30 to 40 seconds .
Replace &apos; name ( ) &apos; with &apos; getMapper ( ) &apos; in GeoShapeFieldMapperTests .
Fixed issue with invalid parse tree shape detection in C + + runtime tests .
Added database commit and compaction to DiskJournalCache
Added &apos; public &apos; modifier to methods &apos; types &apos; in classes &apos; ValidateQueryRequest &apos; and &apos; CountRequest &apos; .
Added ClassLoaderLock registration for Java6PluginClassLoader
Remove import of com.google.common.base.Throwables and replace Throwables.propagate with H2O.unimpl in ValueArray.java.
Added constructor for LineExtensionInfo with TextAttributes parameter .
Added new constructor for RequestBuilder that takes disableUrlEncoding and validateHeaders parameters .
Modified GradleDistributionExecuter.java to correct exception handling .
Removed unused import javax.swing.tree.TreePath ; and simplified getIndexOfChild implementation .
Updated AstroConfig to use Calendar.getInstance ( ) .get ( Calendar.DST _ OFFSET ) instead of tz.getDSTSavings ( ) for daylight savings time calculation .
Added custom evaluators for arbitrary property types in animation .
Added serialization of FailProcessorException to enable proper error handling in CompoundProcessor .
Reduced read timeout and removed unused variable in AjpProcessor .
Remove isString ( ) and isStringArray ( ) methods from UserValue class .
Added import statement for ConsoleEvent and modified Build class to use BuckEventBus for posting build report to console .
Ensure readerIndex of buffer is 0 before async write to avoid JDK writing into wrong region
Updated MongoClientOptions class to use milliseconds for maxWaitTime and connectTimeout parameters .
Removed unnecessary imports and added TextWatcher for layout request on text change .
Added random jitter to ShardCompactionManager &apos;s scheduleWithFixedDelay to avoid overloading database
Added quotes to argument when adding it to list in ExecUtils.java and corrected spelling mistake in RunCommand.java
Added import statement for &apos; ControllerMethodInvoker &apos; and fixed return type check for controller methods .
Removed unnecessary import of java.util.regex.Pattern and updated split method to use String.split ( ) instead of Pattern.compile ( ) .
Update JdbcTokenRepositoryImpl to use &apos; lastUsed &apos; instead of &apos; new Date ( ) &apos; in updateToken method .
Added support for GPU method inlining
Added token ID to intent for task edit fragment
Removed unnecessary search paths in DefaultGroovyPageLocator .
Updated SvnConfiguration to use a new instance of SvnConfigurationState instead of a static variable .
Removed unnecessary &apos; this &apos; references in setKeyedTag method .
Added support for new browsers in Util.java
Added setOnPlaybackStateChangedListener and getOnPlaybackStateChangedListener methods to PlaybackControlsRow class .
Modified CrossOriginFilter.java to update exposedHeaders from exposeHeaders and add a comma separated list of HTTP headers that are allowed to be exposed on the client .
Added check for unboxed type equality in RedundantCastUtil
Added cloning support for Materials in SkeletonControl .
Added logging for missing &apos; from &apos; field in comments
Added support for PPTP encryption
Fixed issue with incorrect width and height calculations in BaseLayoutManager .
Replace Log4j with SLF4J
Allow only doc-level references in CollectInputSymbolVisitor
Renamed constructor parameter in XXX.java
Added method openAtPosition ( String , long ) to S3UnderFileSystem , allowing for opening a file at a specific position .
Fix for issue # 12345 : CropImageView restoration bug
Catching Throwable instead of Exception in ConvertBodyProcessor , SetHeaderProcessor , and SetPropertyProcessor .
Modified EmailAlarmCallback to use effectiveBacklogSize instead of Math.max ( effectiveBacklogSize-1 , 0 ) for subList ( ) method
Added getAppVersion method to Version class and updated MainMenuActivity to use it .
Fix array escaping in PostgresUtils.toPGArrayString ( ) for strings and numbers
Added equals and hashCode methods to Tuple interface
Incremented database version to 51 and added new settings for set install location UI .
Fix issue with OnErrorNotImplementedException and OnErrorFailedException handling
Removed import of ODatabaseObject and added a new check for retaining records in ODatabaseRecord and ODatabasePojoAbstract .
Added import statement for org.jboss.invocation.Interceptor
Replace Delay.force ( ) with Delay.invoke ( ) and FnSeq.first ( ) .force ( ) with FnSeq.first ( ) .invoke ( )
Removed unnecessary Todo comment in OpenSSLEngine.java
Added serverAdmin.deleteDatabase ( ) for remote connection
Added generic type parameter to PaginatedRepository
Added getNotCoveredMessage method and updated getSummaryForNode to use it
Disable doLexingOptimizationCorrectionCheck for better performance .
This commit introduces a new class , LegacyBatchIndexApplier , which reuses the same LegacyIndexTransactionApplier for all transactions in the batch . It also caches the appliers for different IndexCommands for performance reasons .
Removed MyTicketsPage from the list of pages
Added static Main instance and getInstance method
POJOPropertyBuilder : Explicit name check added for renamed properties
Added &apos; --extended-regexp &apos; and &apos; --regexp-ignore-case &apos; options to GitLogProvider.java
Removed unused meta edge builder properties .
Added IBM JDK vendor check in SignatureTests .
Added check for command line in ProcessHandler when process is terminated
Replace C1Chunk with C1NChunk in NewVectorTest
Fixed bug in setSelectedIndex method in OptionPicker class .
Replace user handle from USER _ CURRENT to myUserId ( ) in TvInputInfo
Added permission check for Bluetooth state change
Added flag GRANT _ PERSISTABLE _ URI _ PERMISSION to intent for ACTION _ OPEN _ TREE
Added distinct keyword to JpaOwnerRepositoryImpl.findByLastName query to fetch distinct owners .
Added filtering of available intentions by hint text
Added check for ACC _ BRIDGE modifier in getCovariantImplementation method
HTTP2Session.java : updated process ( ) method to return a boolean value and added a check for closed streams .
Added response content type and expires parameter to OAuth20AccessTokenController .
Added field &apos; mAsyncJournalWriter &apos; of type &apos; AsyncJournalWriter &apos; to class &apos; AbstractMaster &apos; for async journal writes .
Added storedActivity variable and removed unnecessary variable activity
Update DEFAULT _ URI to point to MongoDB instance running on port 27017 instead of 27018 .
Removed unnecessary import statements and simplified code in PyClassImpl.java.
Added mFirstShow variable to track whether the keyguard view has been shown for the first time , and modified the show ( ) method to play sounds only when mFirstShow is false .
Added synchronized method to BuildProgressListenerAdapter
Replace &apos; new Stopwatch ( ) .start ( ) &apos; with &apos; Stopwatch.createStarted ( ) &apos; in StopwatchBenchmark and LocalCache
Added test for empty objects being equal and corrected operation of secure context .
Add hostname to unix-socket URLs for curl in HeliosSoloDeployment
Added Serializable interface to IMethods mock
Added support for OSX fullscreen mode .
Update OtrChatListener to display a user-friendly error message when decryption fails .
Removed unnecessary fields from HystrixRequestEventsJsonStream
Added horizontal scrollbar to console editor .
Added expandRow ( 0 ) method call to InspectDebuggerTree.java and imported Disposable in InspectPanel.java.
Added lambda parameter to GLMBasicTestRegression .
Added IOException handling when closing leader latch and removed unnecessary close call
Fixed PagingPredicate to throw an exception when pageSize is &lt; = 0 and added a reset method to clear anchor map and set iterationType to null .
Added license header and updated package name for StreamRuleResource
Moved scroller from BorderLayout.NORTH to BorderLayout.CENTER in GeneralConfigurationPanel.java
Removed unnecessary System.out.println statements in LightInspectionTestCase.setUp ( ) .
Added parameter @ since 4.1 to method getTicketGrantingTicket ( String ticketGrantingTicketId ) in CentralAuthenticationService interface
Updated Model.java to use generics for Frame return types .
Added unique name check for transport stops
Changed method toBinaryUsingShiftsAndModulus to use while loop instead of for loop .
Removed unused Google Analytics code
Added setWriter method to Json class and modified toJson methods to use it .
Disable reusing task stack views until the visibility bug is fixed .
Added newline character to CantRunException message
Removed unused border image setter method in UIDisplayContainer and fixed constructor in UIStyleBackgroundImage
Removed unnecessary message handling in WebView &apos;s touch event handling .
Added import statement for &apos; edu.emory.mathcs.backport.java.util.Arrays &apos; and removed unused import statement for &apos; java.util.Arrays &apos; in RunMojo.java.
Added prop focusable to package-info.java
CameraDeviceImpl.java : Updated onCapturePartial ( ) to onCaptureProgressed ( )
Minor change in LogoutManagerImpl.java : Changed log level from INFO to DEBUG for logging message related to proxy-granting tickets .
Added @ JsonIgnore annotation to &apos; type &apos; field in Event.java
Added GNU Lesser General Public License ( LGPL ) version 2.1 or later to PostgreSQL93Dialect.java.
Added hack to allow &apos; @ suppress { missingRequire } &apos; to mean &apos; @ suppress { strictMissingRequire } &apos; . Renamed &apos; strictMissingRequire &apos; to &apos; missingRequire &apos; .
Modified FileInfo to use getters for inMemoryPercentage and added tests for pin command
Added private method &apos; thisT ( ) &apos; to class AbstractManagedChannelImplBuilder for correctly typing the builder .
Added import statement for org.apache.flink.annotation.PublicEvolving and added @ PublicEvolving annotation to StoppableFunction interface .
Added import statement for com.intellij.openapi.application.ApplicationManager and updated threadAction method to use runReadAction instead of directly setting classRef.set ( classComputable.compute ( ) ) ;
Removed unnecessary System.err print statement in TaskEditActivity .
Added check for null version in RepositoryLibrarySynchronizer
Removing unnecessary classes from JPAMockSFSBContainerTestCase.java
Added TLS support to OkHttpClient by adding isTLS flag and checking it in the connection specs .
Remove unused method getFilename ( ) in DefaultDiskStorage and mark it as deprecated .
Increased timeout for BluetoothAdapter disable from 5000 to 20000
Remove PDF documentation test
Added check for water and flowing water blocks as soft blocks in BlockUtil.isSoftBlock method .
Renamed &apos; BoundColumn &apos; from inner class to static nested class in CqlPagingRecordReader.java
Removed unnecessary &apos; equals &apos; method call in SelectMessageLocationsPanel .
RevoluteJoint : Changed getLocalAnchorB to use jniGetLocalAnchorB instead of jniGetLocalAnchorA
Update PropertiesComponent.getInstance ( project ) .isTrueValue ( PROP _ COMBINED _ OVERRIDE _ IMPLEMENT ) to PropertiesComponent.getInstance ( project ) .getBoolean ( PROP _ COMBINED _ OVERRIDE _ IMPLEMENT , true )
Moved the creation of the panel for the watches view from the variables view to the toolbar position
Removed unused imports and simplified code in VcsFileStatusProvider .
Added logging statement for SIMPLE ACCOUNT REGISTRATION ... &#91; STARTED &#93;
Fix for SQL injection vulnerability in UseHandler.java by removing unnecessary backticks .
Added getCompactionThroughputMbPerSec and setCompactionThroughputMbPerSec methods to StorageService and StorageServiceMBean
Make tvals field final in TRef
Added cancel button to BreakAddEditDialog and implemented cancelPressed ( ) method to hide add / edit dialogs .
Added FixContextListener and DefaultWebXmlListener to Tomcat.java
Fixed calculation of standard deviations in SoftCluster .
Fixed ServerManagerTest to use await ( ) instead of await ( 25 , TimeUnit.MILLISECONDS ) for queryNotifyLatch .
Added setupTestingRoots ( ) and resetStorage ( ) methods to ActivityTest .
Updated settings : added haptic feedback enabled , live web suggestions while typing search dialogs , backup transport , and setup wizard version ; changed vending PD resend frequency MS ; added vending promo refresh frequency MS and legal terms of service URL .
Throwing IOException with cause for StateStorageException in ComponentStoreImpl .
Added support for multiple values of numResponsesPerArg in ObservableCollapserPerfTest .
Added &apos; equals &apos; , &apos; toString &apos; , and &apos; hashCode &apos; methods to AccessorAttributesCollector and added a limit of 120 useful parameter count .
Added cleanup method to MongoDbClient class
Updated ReplicaSetStatusDomainModelTest to use maxBsonObjectSize instead of Bytes.MAX _ OBJECT _ SIZE
Added PLAINTEXT negotiation type to OkHttpChannelBuilder for grpc transport
Set secondary toolbar visible to false in ConnectionsPane
Added error message with default file path for unset properties in GradleDistribution .
Fixed grammatical errors in LineageMasterClient.java.
Added cleanup method to LaunchPlugin interface and added try-catch block in AppCompiler to call cleanup method
Update to load classes from org.apache.catalina.util instead of org.apache.catalina.
Added support for capsule command-line options with prefix &apos; capsule . &apos; ( # 543 )
Modified BatchConfiguration.java to use trim ( ) method when loading batch.enable property .
Updated SQLTypeMapping to use String class for TEXT types instead of Character class
Added logging statement in HttpConnectionOverFCGI.java for case where channel is not found for a request .
Added import statement for JBPopup and added getHint ( ) method to DocumentationComponent class .
Fixed incorrect list selection handling in DrawerUtils .
Added test for recursive computation with ConcurrentSkipListMap
Removed unnecessary stopUFS method
Added option to disable call history button in search panel
Changed border width from 2 to 1 for drop target component and added empty border for other components .
Fixed SNAC sending for authorization grant in AuthCmdFactory .
Added method getDateModified ( ) to DownloadOsmandIndexesHelper and setLastModified ( dateModified ) in DownloadIndexesThread .
Added method getCollectedCount ( ) and parameter timeout to PacketCollector constructor .
Fixed error message in FileTabCharacterCheckTest to use the correct exception message .
Removed unnecessary imports in RouteBuilderRefTest.java
Removed customization of quick add intent in TagViewFragment and TaskListFragment .
Modified JsonHttpResponseHandler to use parseResponse instead of JSONTokener for parsing JSON responses .
Added import for ToolWindowManager and activated editor component in HectorComponent .
Fixed issue with closing mailbox folders in MailConsumer
Added null safety check for ButterKnife unbinder in onDestroyView method .
Added method exitActual ( ) to PApplet class to allow subclasses to perform custom cleanup before terminating JVM .
Fixed issue with Dijkstra &apos;s algorithm in OSQLFunctionDijkstra.java
Fixed null pointer exception in getHttpRequest method by adding null check for urlString parameter .
Replace &apos; assertTrue &apos; with &apos; assertInstanceOf &apos; in ClientMapStoreTest .
Renamed OilPopulate.java from energy to worldgen
Added locale support for upper case transformation in PagerSlidingTabStrip
Modified QRCodeReader.java to handle cases where the QR code is too close to the edge of the image .
Added support for isolated processes in Activity Manager Service
Added SplitSyncUpOperationTest to OperationTestSuite .
Added logging of thrown exceptions in Http11Processor
Added test case for nested structure in DOMGeneratorTest
Added VDBG logging to NativeDaemonConnector
Added logging of current thread , our dispatch thread , and system event queue to ApplicationImpl.assertReadAccessAllowed ( )
Removed unused import : org.wordpress.android.models.AccountHelper
Added support for multiple editors in pinned mode
Added additional enum values for Greater than , Less than , Equal to , Greater than or equal to , Less than or equal to , and Not equal to in Tokens.java
Remove unnecessary variable and method call in DomInvocationHandler.detach ( )
Added method override for updateTextBounds and changed drawLetter to use floats instead of ints .
Added JOIN keyword to SQLConstants.java
Fixed calculation of spot light angle cosine
Added registration of rest handlers for transport client
Fix issue with incorrect exception message in NettyServerHandler
Added saveSettings ( ) call in SiteSettingsFragment &apos;s onDismiss method .
Fix issue with RequestArguments.java - step &lt; = 0 throws IllegalArgumentException .
Added clone method to RenderTarget class .
Added support for switching playback language based on ISO 3 language code
Removed unnecessary import of LogConfigurator .
Modified ContactListTreeCellRenderer.java to fix the calculation of the width of the status icon and label
Added method updateImportOfElement to PyClassRefactoringUtil , which updates the import statement of an element .
Fixed illegal state error in EditPostActivity by adding try-catch block for unregistering receiver .
Added check for empty topItems list before sorting and outputting recommendations in RecommenderMapper .
Fixed bug in XMPPTCPConnection where shutdownDone was not marked as successful by the writer thread .
Added test for Derby DB creation with DataSourceLoginService .
Added support for sending arbitrary MIME-encoding content via createMessage ( ) and added a new method createMessage ( String ) for sending simple text messages with default content type and encoding . Also , added a new method sendInstantMessage ( Contact , Message ) to send instant messages to a specific contact and added support for registering MessageListeners .
Added methods onShowPreview ( ) , onPreviewShowed ( ) , and onClose ( ) to SlidingLayer class .
FingerprintUnlockController : updated mode handling for bouncer showing
Fixed calculation of average put time in CacheStatisticsImpl
Cache invalidation logic updated in DefaultPersistentDirectoryCache .
Added swipe to refresh feature and hiding empty view when list is not empty .
Fixed illegal argument exception for negative start delay in ViewPropertyAnimator .
Added fallback to ASN.1 DER signature for backwards compatibility in EllipticCurveSignatureValidator .
Added Ratpack Groovy DSL example and updated Java example for using Spring in Ratpack applications .
Added support for custom Smack configurations through reflection .
Added code to log event when default network is lost and there is no replacement
Fixed issue with incorrect canonical text calculation for reference expressions .
Changed JButton icon from &apos; # &apos; to &apos; ⚗ &apos; in Toolbar.java.
Added functionality to CordovaResourceApi : 1 . Helpers for reading &amp; writing to URLs . 2 . Allowed plugins to redirect URLs . 3 . Exposed a way to use OkHttp library .
Added removal of entry from child executors when channel closes .
Fixed rendering of wave underscore and bold dotted line in EditorImpl .
Added support for Java identifier parts in StringUtil.java
Removed unused imports in ClientIssueTest.java
Renamed test method from &apos; testLoadRouteFromXml &apos; to &apos; testLoadRestFromXml&apos;
Update TWLanedLayoutManager.java to set mHorizontalSpacing and mVerticalSpacing to zero by default .
Fixed UnderStoreBlockInStream to use absolute positions within UFS file instead of under store stream .
Added import statement for AgencyAndId and updated TripShort constructor to use AgencyAndId instead of integer for shapeId .
Added PrivateAccess class for worker and block management .
Update InodeTree to use a ConcurrentHashSet for pinned inode file IDs
Updated UndertowLogger to include load balancing group in NodeConfig created log message .
Added logging for exception while getting master ID for txId
Removed unused import &apos; tachyon.worker.ClientMetrics &apos; in BlockWorkerClientPool.java
Fixing error in JavaSourceProcessor.java
HdfsFileInputStream : Remove unnecessary &apos; mCurrentPosition + = ret &apos; line .
Added trace logging for prepared statements with bound terms in QueryProcessor .
Added MIT license headers to MockitoAnnotations.java , Printable.java , and WantedAnywhereAfterFollowingInteraction.java.
Added registration of tick handler with FMLCommonHandler in ModLoaderModContainer
Added option to specify Hadoop version and enabled Google Analytics opt-out by default
Update PlatformDependent class with new logic for determining direct buffer preference .
Added file name parameter to isRestricted method in XIoBridge class .
Removed unused method topLevelSearcher and simplified cache retrieval in FetchSubPhase .
Added logging for unknown packet types in Observer .
Added support for telephony in ConferenceInviteDialog
Added MIME type &apos; application / vnd.espass-espass + zip &apos; for espass files .
AudioPlayer : Ensure player is not null before starting or pausing
Modified StatusBar.java to use RGB _ 565 pixel format and removed hardware acceleration for status bar updates to reduce power consumption .
Fixed issue with decimal format where it would return a Double instead of a Long for integers without a decimal separator .
Removed unnecessary variable &apos; sslProtocol &apos; in Connector.java.
Replace TestDescriptor with Descriptor in DefaultStartEvent.java.
Add pkgLocator and modify setupMockClient , setupMockWorkspaceFiles , createConfigurationFactory in AnalysisTestCaseForJunit4
Added generateParameterAnnotations method to FunctionCodegen.java
Removed unnecessary import of PrintStream and updated outputTo ( ) method to use System.out directly .
Added method toString ( ) to Packet class for Java 8 compatibility .
Removed unused import com.sun.org.apache.xml.internal.security.encryption.AgreementMethod
Fixed SQLServer view naming convention
Added frameCount method to PApplet class .
Added warnings about using timers with long-running treatments and suggested using setTimer instead .
Fixed cut off conference window by adjusting size of parent window .
Added call to buildLayer ( ) in ViewPropertyAnimator.java
MtpStorage.java : Changed reserve space calculation to use long instead of int .
Added import statement for java.io.Serializable and implemented Serializable interface in HostAndPort class .
Replace getPhoneType with getContactType in ContactAccessorSdk5 .
Added support for generating long IDs using a separate collection for storing ID values .
Added import statement for Hex class and modified toString ( ) method to include a hexadecimal representation of the transaction &apos;s identity hash code .
Replace UriUtils with URISupport in SnsProducer
Fixed typo in error message for missing argument ( s ) on rule reference ( ErrorType.MISSING _ RULE _ ARGS ) .
Transaction.java : Changed method isEveryOwnedOutputSpent to return false if any output owned by the given wallet is unspent .
Fix for SonarQube bug # 12345 : Populate analysis UUID column on CE activity
Updated ESTimestampFixup class to use debug logging instead of info logging for scrolling and bulk actions .
Added Apache License header to Foo.java
Added test for ProgressLogger to ensure it logs at least once every second .
ImplementationBodyCodegen : simplified delegation specifier handling
Rename JBossAsCommonConfiguration to CommonContainerConfiguration and update imports accordingly .
Added water.UniqueFrameId to TypeMap &apos;s list of special class names .
Added wildcard path &apos; / * * &apos; to ResourceProperties.paths
VCardParser _ V21 : Changed constructor parameters to use nullable types .
Added test for invalid Windows path on Unix and reserved name under Windows .
Added default case to handle unknown email formats in CursorToMessage
Fix for invalid import due to missing or invalid metadata files .
Added code to delete post ID from ignore list when publishing a draft
Update URL and APP _ ID for ActFmInvoker
Added deprecation warning to BootstrapModule constructor taking ProjectReactor and Object ... arguments .
Added check for node awakeness before attempting heal
Added flag for GBA requirement and updated default values for IMS features .
Removed unnecessary marking of played items as played when skipping episodes and fixed incorrect logic for keeping played items in the queue .
Added ACRA version to User-Agent header in HttpRequest
Added default empty list for controlPoints to prevent null pointers
Added JMX bean for PartitionService with methods isClusterSafe ( ) and isLocalMemberSafe ( )
Remove unnecessary tag addition in Grouper.java
Added check for configBase in getHostWebXmlSource ( ) method .
Added constructor for ExternalClasspathClassLoader that takes a parent class loader as an argument .
Added import statement for NoSuchFileException and caught the exception in PathResource.getRealUri ( ) method .
Added support for remote debugging with _ wx _ devtool parameter .
Fixed Firefox warning for Synctex forward search
Increased time interval for running and cancelling job in MatchJoinCancelingITCase test .
Fixed cache initialization for routes management test
Ensure activity configuration is correct when starting an activity
MulticoreWrapper : Changed call ( ) method to throw Exception instead of catching it and added a try-catch block for better error handling .
Change in UserManagerService.java : Instead of directly calling AppOpsService.setUserRestrictions ( ) , post a Runnable to the handler to avoid RemoteException .
Update icon for PluginModuleType
Added JUnit4 runner to Http2NettyLocalChannelTest .
Added drop method to IndexProperty and updated IndexPropertyMigration to use it
Added import statement for CFMetaData in CommitLogSegment.java
Added null check for ApplicationManager.getApplication ( ) in MockProject constructor and updated PsiNameHelper creation in MockPsiManager .
Fixed type inference for autocasts by using Sets instead of Lists to store type information .
Added retry request sample with whitelisted and blacklisted exceptions
CorruptedFileTest : added check for file name with &apos; write.lock &apos; in Paths.get ( path ) .resolve ( relativeDataLocationPath ) ;
Allow terminal to process keyboard input for non-workbench and non-addin commands .
Fixed incorrect escaping of &apos; world &apos; in HtmlEscaperTest .
, Calypso-style Preference with icon and widget support .
Added import statement for com.intellij.psi.impl.light.LightElement ; and modified the condition for reporting conflicts to include light elements .
Added helper methods for publishing events to MapEventPublisher interface .
Optimize PersistentArrayMap.without ( ) by skipping removal of keys
Removed unnecessary System.out.println statement in AssetBundleReader.java.
Modified PApplet.java to use Thread.currentThread ( ) .getContextClassLoader ( ) .loadClass ( name ) instead of Class.forName ( name ) for loading the applet class .
Added .root _ relative line in PreDexMerge.java for RAW dex store mode
Added category ID &apos; BARBARISM &apos; for words or expressions that are badly formed according to traditional philological rules .
Added getContainer method to AceEditorNative and updated ChunkIconsManager to use it .
Added IOException handling in ZooKeeperConfigurationSource.close ( ) method .
Added spam confirmation dialog and moved spam messages to spam folder
Remove unused imports and update HeartbeatInterceptor to return a non-null value for CLIENT _ HEARTBEAT _ INTERVAL _ IN _ SECONDS
Added deprecation warnings for setAsDeleted and clearMatches methods in Rule class
Remove unused parameter &apos; contentType &apos; from IonRequestBuilder.setStreamBody ( ) and RequestBuilder.setStreamBody ( )
Added import statement for &apos; com.intellij.openapi.actionSystem.PopupAction &apos; and implemented &apos; PopupAction &apos; interface in &apos; ShowPopupMenuAction &apos; class .
j2objc : Use a new object rather than this to avoid reference cycle in Reader and Writer constructors .
Added support for generating ActionBarDrawerToggle automatically when creating a Drawer .
Added support for initial vertex tangents in VertexBuffer
Replace InternalLoggerFactory.getInstance ( Class &lt; ? &gt; clazz ) with InternalLoggerFactory.getInstance ( String name )
Modified WsSession.java and WsWebSocketContainer.java to change protected void expire ( ) to protected void checkExpiration ( )
Added support for resuming broadcasts in AbstractReflectorAtmosphereHandler .
SettingsCmd : updated documentation for system user
Added import statement for TimeoutException in MoreExecutors.java
Added method totalUsed ( ) to VirtualMachineMetrics
Fixed spelling mistake in FileDirContext.java
Updated ForeignKey.java to use StringUtils.StringUtilsFormatter for formatting foreign key and primary key columns in toString ( ) method .
Fixed exception handling in ReflectionUtil
Update AppearancePreferencesPane.java to use plot ( ) function with correct arguments .
Added avatar visibility toggle based on settings
EditAccountActivity : Show or hide &apos; Clear devices &apos; and &apos; Change presence &apos; buttons based on Omemo support .
Modified CoordinatorLayout.java to implement onDependentViewChanged ( ) method for nested scrolling behavior .
Remove unused import and main method
Fixed issue with refreshing children files in LocalFileSystemImpl .
Added import statement for java.nio.file.Path and updated getCurrentSaveRootDirectory method to use Optional &lt; Path &gt; instead of File .
Updated CookieAuthenticator to use encryptor.decrypt ( ciphertext ) instead of new String ( encryptor.decrypt ( ciphertext ) , UTF _ 8 )
Added check for non-existent names file and updated index invalidation logic .
Modified AbstractGrailsArtefactTransformer.java to replace generics placeholders and update the way the API instance is accessed .
Moved methods to a deeper module and made getters and variables inline
Added getArguments and setArguments methods to ShadowFragment .
Added logging statement for installed atmosphere interceptor and added disable defaults option in XML
Replace AudioNotifierImpl with AudioNotifierServiceImpl
Added placeholders for user-created terminal sessions in Commands.java and TerminalPopupMenu.java
Modified ConnectPlugin.java to return the active session state instead of erroring out when no valid session is found .
Fix unreferenced switch data block in CodeItem
Added import statement for android.util.Log and added log method to NetworkMonitor class .
Remove check for slf4j implementation presence
Added setConstraintMappings ( ConstraintMapping &#91; &#93; constraintMappings , Set &lt; String &gt; roles ) method and updated existing methods to use Arrays.asList ( ) instead of specifying explicit arrays .
Added email follower column to PeopleTable and updated getFollowers method to include is _ email _ follower filter .
Added a blank header to the listView with a fudge factor to account for the info container - global layout listener below will use the actual container height once it &apos;s known . Also , added tagging of the spacer so we can identify it later .
Added method to clear SSL certificate lookup table and updated WebViewCore to use it .
Added semicolon to $ Domain cookie attribute in CookieUtil.java and updated TestCookieUtil.java to include $ Port , $ Path , and $ Version attributes in cookie decode test .
Added method &apos; find &apos; to class ContainerUtil for finding an element in an iterator that matches a given condition .
Added field &apos; bufferSize &apos; to class FixedRecvByteBufAllocator and updated newHandle ( ) method to return a new HandleImpl instance with the specified buffer size .
Update DefaultGrailsControllerClassTests to use correct view name
Fixed issue with expandOrCollapseRoots ( ) method in VcsLogGraphTable.java
Fixed hack for testing remote repository existence in GitCloneDialog
Added adjustLayourParams ( ) call in onLayout ( ) to update layout parameters when forced .
Fixed bug in SuperColumn.java where marked columns were not being properly considered for deletion
Modified CachedResource.java to use DirectBufferCache instead of cachingResourceManager.getDataCache ( )
Added handling for visibility changes during layout transitions
Replace type check with named method strategy
Added support for reading config flags with sizes greater than 64KB .
Modified MultiMapService.java to conditionally clear migration data based on migration endpoint
Added null check for Comments instance in Delombok.java
Added timeout parameter to UdpClient.receive ( ) method and updated NioDatagramChannelTest to use it .
Added null check for service descriptor in Requirements.java
Updated regular expressions for parsing task outputs .
PipelineInstanceModel.java : Removed unnecessary method abbreviate and simplified getRevisionOfLatestModification ( )
Added logger for debugging purposes in NativeBinariesModelPlugin .
Trash remote messages marked as trashed locally
Removed unnecessary constructor and updated the remaining constructor to use a generic keyed object pool configuration .
Added method findFirst to CollectionUtils , which returns the first element in an iterable that matches a given filter , and also added a convenience method for filtering a set .
Added shard _ id and shard _ uuid to IndexInserter
Added import statement for UnstableApi in Http2ServerDowngrader.java
Added measurement and layout for children views in TWAbsLayoutManager .
Removed unnecessary import of &apos; SortedList &apos; in MultiValueMode.java.
WebEditorOptions : Enable sync tag editing and tag tree highlighting in non-unit test mode .
Replace DropwizardRule with DropwizardClientRule in DropwizardClientRule.java
Update NetworkStatsService to correctly calculate total bytes and packets for mobile and wifi samples .
Added import statement for android.annotation.NonNull and added methods getOutline and getConstantState to InsetDrawable class .
Increased execution timeout for HystrixObservableCollapserTest from 50 to 1000 milliseconds .
Added setting for showing sync icon in PhoneStatusBarPolicy
Fixed bug in isDomainTrusted method in TrustedSslDomainTable.java
Added boolean variable &apos; handled &apos; to keep track of whether the event was handled by the actor in the scene .
Update FlexibleDividerDecoration.java to use getChildPosition instead of getChildAdapterPosition
Updated JBoss XSD filter to include WildFly in AbstractValidationUnitTest .
Remove unused methods getReplicaIndex and getPartitionId from Invocation and PartitionInvocation .
Added &apos; lastKey &apos; field to AntiEntropyService and updated add method to keep track of last processed key
Renamed method getUrl ( ) to getUrl ( ) and removed unnecessary variable name in IServer interface .
Removed unnecessary methods and simplified code in JsToStringGenerationVisitor .
Update PluginJetFilesProvider to use isInSourceContent instead of isInContent
Removed endpoint &apos; activiti &apos; from the list of endpoints to be proxified by spring boot admin .
Added import statements for File , FileInputStream , and FileNotFoundException , and updated parse method to throw FileNotFoundException .
Throw CacheNotExistsException with cache name instead of just &apos; Cache is already destroyed or not created yet &apos; .
Added a check to ensure the drawer is not null before resetting the drawer content .
Added small inserts and semantic similarities to string similarity test .
Update copyright year in AutowiredAnnotationBeanPostProcessor.java and change required status determination for annotations without a &apos; required &apos; method .
Removed unused variables and methods in AlertDialogWrapper .
Replace HashMap with ConcurrentHashMap in ClassNameUtil
MccTable : Update Netherlands Antilles MCC code from &apos; nl &apos; to &apos; ai &apos; .
Added shutdown method call in Neo4jGraph constructor to ensure that rawGraph is properly shutdown in case of exceptions .
API : use &apos; responseStatus &apos; instead of STATUS _ OK in getDefaultResponseHeader
Modified ContentProviderStoreBase.java to change abstract protected Uri getContentUri ( ) to abstract public Uri getContentUri ( )
Added functionality to TestsPresentationUtil to handle test names that start with a parent name , and also added a comment to the parent test proxy to clarify the naming convention .
Update import statement for ClassUtils from flywaydb to spring
Added new method &apos; invokingWithException &apos; to interface InvocationListener , and updated the existing &apos; invokingWithReturnValue &apos; method to include a new parameter &apos; locationOfStubbing &apos; . Additionally , updated the documentation to reflect the changes .
Modified SparkLinesRenderer to use a smaller maximum dimension for charts ( was 5000 , now 3000 ) .
Modified XMLLightweightParser.java to fix buffer size issue .
Added stream caching details to ContextInfo
Debug logging for created files
Added support for people in notifications on KitKat devices .
Added import statement for PsiTreeUtil ; updated code to use PsiTreeUtil.getParentOfType instead of PsiElement.getParent ( ) to get the parent PsiJavaFile .
Added support for Explorer tests in Generator.java
Update ProgressBar.java to use getCurrent ( ) instead of mCurrentDrawable
Removed unnecessary call to getGL ( glDrawable ) in PJOGL.reshape method .
Update license headers for LoginServlet.java to reflect change in copyright year .
Renamed variables for better naming consistency and added a new method getSceneFinalRawData ( ) to retrieve the content of the color buffer of the FBO &apos; sceneFinal &apos; .
Added import statement for ByteArrayOutputStream and modified code to use ByteArrayOutputStream instead of ByteOutputStream for serialization .
Added checkClass ( ) method call in OrientEdge and OrientVertex remove ( ) methods .
Replace first occurrence of $ 1 with $ 1 in getPlaceholderText method .
Removed unnecessary import of Log class in Scheduler.java.
Added support for non-model columns in ForeignKeyColumnDefinition
Added constructor for servlet class in ServletHolder
Added flag Intent.FLAG _ RECEIVER _ REGISTERED _ ONLY _ BEFORE _ BOOT to sendBroadcastToAll method
Added version _ code property to AnalyticsTrackerMixpanel and updated registerSuperProperties method to include it .
Added toString ( ) method to BsonBinary class .
Added missing simple class name in assertPermitted ( RuntimeException ) method .
Fixed bug in TUGIContainingTransport.java by returning the correct value .
Update DebugInfo.java to adjust line number calculation for better accuracy .
Modified DefaultTraceEventMessageTest to use File.separator instead of hardcoded path separator .
Renamed static ExtensionPointName to final ExtensionPointName in AbstractProjectViewPane
Remove unnecessary System.out.println statement in TranslationProcessor .
Added methods getCookies ( ) and getExtensions ( ) to WebSocketClient class .
Added fallback implementation for onSearchRequested ( ) in PhoneWindow
Suppressed nullability warnings for Pair.getFirst ( ) and Pair.getSecond ( )
Added &apos; value &apos; annotation to DependedUpon interface and added &apos; qualifier &apos; field to Resource class .
Added private constructor to Charsets class for GWT compatibility
Fixed issue with empty resources in BinaryGrailsPlugin
Added assertion for isFreed ( ) in PooledByteBuf.suspendIntermediaryDeallocations ( )
Catching Throwable instead of NoClassDefFoundError in DefaultUdcInformationCollector
Removed &apos; final &apos; modifier from MessageSettings class .
Added import statement for OResourcePool and removed unnecessary import statement for Map .
Added test cases for date field operations and number as boolean
Modified ExampleRunner.java to print the result of the read-config-as-xml operation instead of the entire request .
AstHist.java : Restricted Hist to non-constant numeric columns .
Fixed issue with big text not being displayed in notifications
Added debug statements for postProcessDependencies in EnglishGrammaticalStructure and fixed an issue with FileBackedCache &apos;s merge method .
Added method toFullString ( ) to class Version
Added support for Eclipse project &apos;s module-related roots and libraries
Changed ioThreadMultiplier from 8 to 2 in AsyncHttpClientConfigBean.java
Added rewind ( ) method call to stencilBuffer in PGL.java
Added test for secure Redis connection pooling with authentication .
Added serialVersionUID to ExtractorSpanishAuxiliaryTag and ExtractorSpanishSemiauxiliaryTag
Updated GoogleAccountsServiceTests to use Pattern.compile with a raw string literal for the NotOnOrAfter attribute .
Updated join clause in JdbcDatabaseSnapshot.java to use sys.indexes instead of INFORMATION _ SCHEMA.TABLE _ CONSTRAINTS
Added DEPRECATED annotation to CreateFrameHandler and updated navbar links for CreateFrame and ImportFiles endpoints .
Added test for SourceMapConsumerV3 with map and sources with root .
Disable storage and network timeouts in OConsoleDatabaseApp .
Removed &apos; final &apos; modifier from Transformation class .
Added mShowWhen = true in NotificationCompat.java.
Replace getCell ( ) method with get ( ) method in BinningClusterer.java
Added parameter &apos; includeLibsJarsFromApklib &apos; and &apos; includeLibsJarsFromAar &apos; to AbstractAndroidMojo with default value as false
Updated print statement for better readability .
Fixed issue with EventDispatcher.compareEvents ( ) method .
Added import statement for OnScrollListener and updated reportScrollStateChange method calls to use SCROLL _ STATE _ TOUCH _ SCROLL and SCROLL _ STATE _ IDLE
Changed image scale type from EXACT to IN _ SAMPLE _ INT in ImagePagerActivity .
Remove unused import statements in TachyonWorker and BlockWorkerHeartbeat classes .
Removed unnecessary frameBuffersManager argument from SimpleBlendMaterialsNode.initialise ( )
Added logic to setCurrentItem based on mode and updated setMode method to update mCurItem accordingly .
FsTranslog : removed unnecessary ThreadLocalRandom usage and simplified logic for selecting the location of a new translog file .
Added constructor for SharedPreferencesProvider to allow for explicit context and context provider injection .
Increased warmup messages from 10000 to 12000 and runs from 200000 to 240000 for better performance .
Removed unused method getCompletionShortcutText ( ) and removed unnecessary import of KeymapUtil .
Added translations version to debug output .
Added method putAAA4 ( int &#91; &#93; &#91; &#93; &#91; &#93; ) and putAAA8 ( long &#91; &#93; &#91; &#93; &#91; &#93; ) to AutoBuffer class .
MailRecipientsTest.java : Updated SMTP URL scheme from &apos; smtp &apos; to &apos; smtp : / / &apos; for better compatibility .
Removed unnecessary &apos; return &apos; statement in AlertMessage.getMaxVer ( )
Adds a local site ID to the top of list of recently chosen sites .
Added workarounds for MathJax sizing on Windows .
Added Serializable interface implementation and updated serialVersionUID for CompactStringObjectMap
Fixed bug in MUCRoomImpl.java : updated crJID construction to use null instead of empty string for resource part .
Remove tool window manager listener
Fixed worker client heartbeat executor to correctly handle periodical heartbeats
Added userAgent method to IonRequestBuilder and RequestBuilder
Fixed debug logging for packed-refs lines in GitRepositoryReader .
Disable walk limit and only search for one itinerary when transit is not used .
Added default implementation for GET requests
Incremented revision version from 0 to 1 .
Update copyright year in ExtensionPoint.java and ExtensionPointImpl.java
Added check for stop status before initiating stop process in DefaultMessageListenerContainer .
Added support for unauthenticated principals in SimpleSecurityManager
Added HTTPS support to EntityParser by adding &apos; https : / / &apos; protocol check .
IndentationCheckTest.java : updated throw message for inconsistent comments
JDBCRealm : Use try-with-resources for PreparedStatement and ResultSet
Update NotificationPanelView to use mQsDensityContainer instead of mQsContainer for layout parameters
Update Random seed from System.currentTimeMillis ( ) to System.nanoTime ( )
Fixed issue with PostgreSQLProjectCreator.java
Added import statements for HttpURLConnection and SocketTimeoutException , and updated getContent method to use HttpURLConnection instead of URL.getContent ( ) to handle timeouts .
Added support for resource change listeners and updated error messages .
Added support for different screen orientations in SearchPOIActivity .
Added OOM test
Update log path in PerformanceWatcher # deleteOldThreadDumps
Added null check for nextMessage in Conversation entity
Added layout manager to TabPanel and changed scrollpane position
Fixed typos in ExecutionHandler.java
Fix bug in EmojiconsFragment : Implement OnEmojiconBackspaceClickedListener interface check
Fixed alert dialog show method in WelcomeActivity .
Fixed issue with generating closures for local class constructors
Added exclusions for Google and Apache classes to prevent issues with Optional and CauseTracked events
Added Apache License header to FileDownloadFlowThreadPool.java.
Added &apos; storage / transaction-success-callback-isolated-world.html &apos; to ignoreResultList in FileFilter.java.
Fixed evaluation of candidates in EvalMapperTest
Added tearDown method to CrateClientTest to close the client and call super.tearDown ( ) for proper resource cleanup .
Updated GetViewDefinitionGeneratorMSSQL to use sys.objects instead of INFORMATION _ SCHEMA.VIEWS to retrieve view definition
Remove legacy attribute map support in SkylarkCallbackFunction
Added mIsFragmentPaused flag to handle fragment pause state
Fixed grammatical error in UtilityCommands.java
Removed import of Executors and added import of TestScheduler in Scheduler.java
Replace NoneShardSpec with schema.getShardSpec ( ) in RealtimePlumber .
Added support for ShaderNodeDefinitionLoader in AndroidAssetManager .
Added tests for Option.none ( ) .orElseThrow ( ) and Option.toOption ( )
Added null checks for marshaler annotations in Bro class .
Modified ProguardMojo.java to use project.build.directory instead of project.build.outputDirectory for output directory path
Fixed capitalization of single-character strings in StringUtil.capitalize ( )
Update TestNodeInfoMapper to use PluginState.STOPPED instead of PluginState.INSTALLED for plugin info
Removed synchronized block in XmlConfiguration.initParser ( ) method .
Added methods &apos; merge &apos; and &apos; swap &apos; to RemoteConference class , and updated &apos; hold &apos; method to use mConnectionService.hold ( mId ) ;
Added check for mID ! = View.NO _ ID in findViewInsideOutShouldExist and findViewByPredicateInsideOut
Removed unused method isCursorInRMode and simplified getSourceDocumentPath .
Added method createCacheConfig ( ) to CacheClusterWideIteratorTest .
FMLRelauncher : updated setupHome method to set minecraftHome and log version information
Added animation to shifting magic method
Replace QueryBuilder filterBuilder with queryBuilder in AliasAction.java
Added action listener to text field in GitSwitchBranchesDialog
Removed unnecessary import of ManyToOneRingBuffer .
Added @ Generated annotation to MetaAttribute enum in JavaGenerator and MetaAttribute classes .
Disable XMPP notification service when app is in background .
Fixed package clean actions
Removed hack for CIDR support in LocalInspectionsPassFactory .
Added SonarQube copyright and license headers to SetAssigneeTest.java
Removed unused import and Log statements in AbstractChartView.java
Removed unused methods get ( InetSocketAddress ) and get ( InetSocketAddress , boolean ) in TachyonFS class .
Added soft input mode for hidden keyboard in NotificationsDetailActivity
Added check for repository.hasCommits before accepting ticket updates
Fixed user rotation mode and accelerometer default settings .
Changed comparison of podcast titles to use lowercase for case-insensitive sorting .
Fixed scaling issues by adjusting the calculation of sourceRatio and targetRatio .
Added check for nlambdas = = -1 and set default value to 100
Fix for KotlinSafeDeleteProcessor.isInside ( ) method to correctly handle Kotlin light methods .
Removed unused variable ATTRIBUTES _ CACHE _ KEY and updated cache expiration duration from 2 hours to 2 days .
Added support for normal maps in TextureAtlas .
Update Row.java to use variable column count in RowBuilder constructor .
Removed unused variable &apos; myModel &apos; in ProjectImpl .
Added validation for empty fragments in DiffFragmentImpl and LineFragmentImpl .
Update DarculaTextBorder.java to change insets for search fields with history popup and without history popup .
Added crash reporting for OutOfMemoryError in ImageHelper .
Added check for parent directory existence before creating wrapper JAR file .
Added error handling for missing static fields in analyzeSymbol method .
Disable minimum size honoring for splitter in InspectionResultsView
Remove unnecessary &apos; throwImmutable &apos; method and simplify &apos; nativeNullifyLink &apos; method .
Added progress indicator for synchronizing output directories in CompileDriver .
Renamed variable &apos; hj &apos; to &apos; hl &apos; in method getArray ( ) of class ServerInterface .
Added assertion for HikariDataSource toString ( ) method in ShutdownTest .
Renamed class BikeParkUnlinked and updated import statements to reflect new package structure .
Removed unused fields from ACRAConstants : SETTINGS _ SYSTEM , SETTINGS _ SECURE , and SETTINGS _ GLOBAL
Removed unused attributes and deprecated annotations in Globals.java
Modified SwipeLayout.java to change the condition for determining the current direction index when swiping .
Fix NullPointerException in equals ( Object ) when comparing with empty messages
Added mutation method for DrawableContainer
Removed unnecessary override of run ( ) method in AsyncChannel .
FMLLog : Changed order of parameters in log method to pass exception as second argument instead of last .
Bump UndertowExtension &apos;s ModelVersion from 2.0.0 to 3.0.0
MixedBulkWriteOperation : Changed type of &apos; writeRequests &apos; from &apos; List &lt; WriteRequest &gt; &apos; to &apos; List &lt; ? extends WriteRequest &gt; &apos;
Added code to grant default permissions for all users .
Remove unnecessary &apos; isNew &apos; flag in WPActionBarActivity .
Fix illegal argument exception for null event in WebViewInputDispatcher
Added import statements for ApplicationManager and ApplicationEx , and added exit method call to ApplicationEx in TraverseUIStarter .
Added password encryption for GithubSettings
Added &apos; sandbox &apos; attribute to PreviewFrame element to allow scripts in HTML preview
Removed unused method &apos; prepend &apos; and simplified &apos; append &apos; method in ByteBufferPool .
Added sendInitialPresence ( ) method call in ConnectionConfiguration and updated login method to use it .
Enable incremental compilation and separate daemon process for JavaCompile task .
Renamed Pair to Couple in LoadingOrder.java
Added getAppearanceController method and fired PropertyChangeEvent with correct object reference .
Set map store implementation using writable config from node engine .
Added test cases for ERB scope in multiline query condition and piped query .
Added support for AtmosphereResourceFactory and configured it to use the new class instance .
Removed unused import statement for RomanianDiacriticsModifier .
Fixed incorrect Y-coordinate assignment in WrapGraphics.java
Removed unused annotation &apos; Print &apos; from RebufferingInputStream.java
BytecodeAnalysisIndex : enabled by default
Modified ColorSpaceTransform.java to use COLUMNS instead of ROWS for indexing mElements .
Removed unnecessary imports and updated import statements for NewsActivity.java
Fixed incorrect error message in AbstractAccessibilityRecordAssert when asserting Y scroll value .
Added method getRequest ( ) to AsyncHttpResponse interface and implemented it in AsyncHttpResponseImpl class .
Parse card and device numbers using try-catch blocks instead of Integer.parseInt ( ) to handle invalid input .
Added logging for SQL statement cache warnings and added debug statements for cache inspection .
Update CheckSum.getCurrentVersion ( ) return value from 2 to 3 and standardize newline characters in StringChangeLogSerializer .
Simplified KeyParser.toString ( ) implementation .
Remove unnecessary &apos; lastConnectionFailure &apos; variable .
Removed import of ArrayAdapter and java.util.ArrayList ; , removed unnecessary code , and implemented ObservableScrollViewCallbacks interface .
Allow VoldemortAdminTool to run in batch mode by default , and add --auto option to enable / disable this behavior .
Update PowerNotificationWarnings.java to cancel notifications for invalid charger and saver modes .
Added type erasure for declaring type in MethodGraph &apos;s extendBy method .
Fixed LocalProperties.areFieldsGrouped ( ) to properly handle null groupedFields .
Disable HW layers for animations on ICS to avoid crashes
Flip transit mode logic in AnalystWorker .
Modified DistributedExecutorService.java to change the order of operations in the &apos; executeOnNode &apos; method .
Added import statement for &apos; com.intellij.psi.templateLanguages.TemplateLanguageFileViewProvider &apos; and updated code to use it instead of &apos; JspFile &apos; .
Removed unnecessary SessionTransactionAsyncCallback creation in InOnlyMessageHandler .
Added mocking for FileReader and BufferedReader in InstanceProfilerTest .
Added logging of SDK version in PrivacyManager when missing method is encountered .
Added getReachedVertices method to return a map of all reached vertices and their distances from the origin .
Fixed XML reader to handle exceptions properly .
Added thread factory to DhnsGraphController &apos;s event bus
Added flag INCLUDE _ NOT _ IMPORTANT _ VIEWS to AccessibilityManagerService.performAccessibilityAction
Added project reference to FileBasedIndex
Added support for getting the assigned value in PyAssignmentStatementImpl .
Added Apache License , Version 2.0 to MyMapFragment.java
Added note about thread-safety in BlockStoreContext and FileSystemContext
Renamed KafkaInput logger to RadioKafkaInput logger
EchoClient : Remove unnecessary EventLoop import and configure client using ChannelBootstrap .
Added import statements for net.minecraftforge.fml.relauncher.Side and net.minecraftforge.fml.relauncher.SideOnly , and added @ SideOnly ( Side.CLIENT ) annotation to getGuiSprite ( ) method .
Fix buffer overflow in ProcessStats.java
Added method idnEquals for comparing domain names in a case-insensitive manner , and added Arrays.equals ( ) method to compare byte arrays .
Fix for hiding 2 extra params for a macro in Compiler.java
Change stream version from byte to int in AssetBundleHeader.java
Renamed getContextAttributes ( ) method to return a Map &lt; String , Object &gt; instead of a Map , and updated the corresponding Javadoc .
Modified StandardContext.java to change resourcesStart ( ) method from boolean to void and removed unnecessary code .
Fix plugin removal error message to use &apos; elasticsearch-plugin &apos; command
Allow read access to highlights in editor document
Update copyright headers for QuadItemTest.java and StaticClusterTest.java from Stefanos Togoulidis to Google Inc .
Fixed a bug in FixedLengthStreamSourceConduit.java where exitRead was not correctly handling the case where transferTo returns -1L
Added nsubj ( eat-5 , dog-2 ) dependency in LexicalizedParserITest .
Added missing &apos; endObject &apos; in RestPendingClusterTasksAction and RiversService
Fixed assertion in RuntimeOptionsTest to compare pattern correctly .
Added check for security access when adding clusters to list of readable clusters .
Added logging for branching detection in CompareTxStreams
Added error handling for normalization in IndexQueue
Added option to filter classes by name
Fix for EVENT CANCELLED ! message in MixinItemInWorldManager
Added delete user and repository functionality to GitBlit .
Added import of Retrier.Hint ; and changed Retrier.retry ( ) to include Hint.CRITICAL for critical index creation
Update AbstractRegisteredServiceTests to set anonymous access and description for registered service .
Disable autocorrection for SuggestionAutoCompleteText
Added support for application / vnd-com.apple.pkpass MIME type .
Clear missing values in NewChunk
Added support for selecting null type in GrTypeComboBox
Added detailed message and throwable to ConnectionClosedException constructor .
Update ProtoSchemaParser.java
Renamed file from src / com / interview / array / AllAdjacentCombination.java to src / com / interview / recursion / AllAdjacentCombination.java
Added support for parametric types and updated connector factory registration
Change type field from canonical name to name in SerializableNativeValue
Added method getName ( ) to GroovyTestCase to make test cases look more like AgileDox , and updated assertArrayEquals ( ) method to use InvokerHelper.toString ( ) for better output in case of failure .
Added method createSnapshotFor ( Engine engine ) to TxEngine class .
Added parameter for filtering by component UUID to QueueWsAction .
Remove unnecessary ImmutableList.copyOf ( ) method call in AndResponseHandler .
Added Apache License header to IOSelectorOutOfMemoryHandler.java
Added throwArity method to AFn.java with improved error message
Fix for updating test execution table
Added clearing of Facebook cookies for https domains .
Disable console logging for exceptions during shutdown
Added @ Deprecated annotation to unused MapMessage constructor
Added warning log for invalid texture directive and added handling for tex _ address _ mode
Added code to refresh time source when stale in NetworkStatsService
Added visitReferenceParameterList method to RefJavaManagerImpl to handle type references in parameter lists .
Removed unused configuration options and simplified HBaseStoreManager
Added null check for empty phone numbers in PhoneNumberUtils.normalizeNumber ( ) method .
MediaSourceWPImages : Changed fadeInImage to setImageBitmap for better performance .
Modified Runner.java to print error and input strings for debugging purposes .
Added check for provider.acceptFile ( inputData.getFile ( ) ) before getting include infos
Increased sleep time in HystrixCommandMetricsTest from 25ms to 150ms .
Added import statement for ProjectFilesystem and added a new constructor that takes a ProjectFilesystem object as an argument .
Added &apos; setEditBeforeRun &apos; method to RunnerAndConfigurationSettingsImpl
Added debug statements to JSFModuleIdFactory
Fix spanOrQueryBuilder to allow empty clauses
changed getAspectRatio ( ) to mI.getAspectRatio ( )
Added setScriptBaseClassFromConfig ( ) call in ModuleNode constructor
Added support for ON DUPLICATE KEY UPDATE clause in InsertOnDuplicateStep and InsertReturningStep
Fixed type calculation for GrClosableBlockImpl
Fixed bug in AppWidgetService : now properly handles ACTION _ EXTERNAL _ APPLICATIONS _ UNAVAILABLE intent
Added helix manager instance config change listener to helix broker starter .
Added test for parallel edges in directed graphs
Fix for Java import order check
WebSocketImpl : removed unused variable and updated pause and resume methods to use the underlying socket &apos;s pause and resume methods .
Replace ArrayList and HashMap with synchronized collections for better concurrency .
Update LocalLibraryDependencyResolver to use BinarySpec instead of BuildTask for build dependencies
Flip topNMetricVal comparison to match Druid &apos;s expected behavior
Removed context name from optionalMBeanNames method
Renamed method from &apos; findField &apos; to &apos; findFieldOffset &apos; in UnsafeAioChannelFinder
Fixing illegal state exception for creating class inside transaction
Removed unused import statement for &apos; List &apos; in IndexMergerTest.java
Allow &apos; style &apos; and &apos; object &apos; elements in &apos; head &apos; tags .
ConfigTest : updated isPostgresLike ( ) to isDerbyLike ( )
Modified EurekaInstanceConfig.java to return secure port instead of non-secure port .
Replace Skin constructor with Gdx.files.internal ( &quot; data / uiskin.json &quot; )
Added import for KotlinFrameworkDetector in KotlinMavenConfigurator
Removed NeuralNetIrisTest and NeuralNetSpiralsTest from AllTestsSuite .
Increased sleep time in RecursiveWatcherTest from 200ms to 400ms and from 400ms to 800ms to allow for proper watch service setup and event firing .
Added clearDisplayList ( ) call in destroyHardwareResources ( ) method .
Increased IN _ PROCESS _ RESCHEDULE _ BATCH _ SIZE from 100 to MAX _ IN _ PROCESS _ PAGES
Set stamp binaries option to false by default and add --stamp option to BuildViewTestCase .
Toggle preference for showing diagnostics in R and C + +
Updated copyright year and added Windows native look support .
Added file length calculation for block size in BlockMeta constructor .
Updated DistanceCalculatorPlugin to display distance measurement mode
Added imports for ManagementEdgeID , ManagementGroupVertexID , and ManagementVertexID to ManagementTypeUtils.java.
Changed result of task deletion from CANCELED to OK
Renamed BRICK _ BAR to BRICK _ BAR in ItemType enum .
Added name prefix for bike parks
Added &apos; final &apos; keyword to Tuple3 and Tuple4 classes
Edit comment API call updated to use XMLRPCFactory instead of ApiHelper
Added inDictionaryStream field to SliceDictionaryJsonReader and updated its initialization logic .
Fix race conditions in releaseCameraInstance by removing camera from map before releasing .
Changed FSCache initialization in CompileContextImpl to use NO _ CACHE for project rebuilds .
MocoException.java : Removed unnecessary constructor and updated the remaining constructors to use &apos; final &apos; keywords .
Removed unnecessary logger statement in PerformParallelRequests.java
Added ClusterSessionListener to avoid having an empty cluster with DeltaManager
EntityMapper : Prioritize constructor TTL value over class-level @ TTL annotation
Removed &apos; Document commit thread &apos; from the list of well-known offenders in ThreadTracker .
Validate client ID in TokenEndpoint
Fixed incorrect projection axis calculation in Intersector.java by using modulo operator for indexing vertices array .
Remove unnecessary &apos; toString &apos; method in DetailAST
Update ThemeBrowserAdapter to set default image resource for screenshot when URL is null , and update RestClientUtils to use new format for getting current theme .
Replace &apos; value &apos; with &apos; number &apos; in IPAddressMap.java
Added import statement for ATN and created constructor for RuleContextWithAltNum .
Added test for ShiftRightLinesAction in visual block mode .
Added mesh update counts and bounding box updates in GeometryBatchFactory .
Added setLinearDamping and setAngularDamping methods to PhysicsRigidBody , and removed unnecessary constructionInfo field
Added support for free version of SRTM plugin .
Added support for copying concatenated strings to clipboard
Update OrderedAdapter.java to use &apos; since &apos; instead of &apos; see &apos; in Javadoc comment
Added nullability annotation to TypedArray.getDrawable ( ) method .
Removed unused imports : File and URI
Renamed PackagePrivateAccess constructor parameter from &apos; elementPackageName &apos; to &apos; packageName &apos; , and removed unused variable &apos; helperUsedMethodMap &apos; . Also , updated the package name in the className from &apos; com.raizlabs.android.dbflow.processor.definition.column &apos; to &apos; com.raizlabs.android.dbflow.processor.definition.column.PackagePrivateAccess &apos; . &apos; ,
Rename FallbackTypeConverterDoesNotThrowExceptionTest to FallbackTypeConverterShouldNotThrowExceptionTest
Added checkSupertypes : YES to findMethodWithTranslatedName in IosMockMaker.java
Added support for resolving dependencies with changing versions using Ivy &apos;s ChainResolver and PatternMatcher .
Added log statement for non-multicast capable interface and removed unnecessary throw statement
Added check for concrete values in object mapping
Fix checkArgument message for k in Ordering.leastOf
Removed deprecated notifications and updated color scheme .
Fix issue with incorrect error message in StressProfile.java
Throwing parameter exception as runtime exception instead of wrapping it .
Added @ SuppressWarnings annotation to LeftCurlyCheck.maxLineLength field
Added show title to UpcomingFragment and updated episode number formatting .
UpgradeService : removed GPS permission , fixed subtasks and crash issues , and improved voice add functionality .
Added missing &apos; } &apos; at the end of DelegatingFuture.java
Removed duplicate key checking and simplified get ( ) method in ExtPropertiesLoader .
Removed HtmlDecoder.decode ( value ) in Attribute constructor
Added null checks for street and backStreet in OpenStreetMapGraphBuilderImpl
Replace params with binaryParams in createAppleDsymForDebugFormat
Increased crash test timeout from 1 minute to 0.5 minutes to allow for slower disk drives .
Added parameter &apos; mimeType &apos; to GsonConverter constructor and updated its usage throughout the class .
Added test for unsubscribeWhenNotSusbscribed ( ) method
Added loadPostPreview method and updated PreviewPostActivity to use it
Update RouteProvider to use COMPLEX routing mode for derived routing from CAR
Fixed issue with null artifact file handling in DefaultArtifactResolutionCache # storeArtifactFile .
Added thread interruption check in AppCompiler.compile ( ) method .
Removed imports for Modifier.ModifierKeyword and PrimitiveType , and removed unused import Type .
Added settings.isIncognitoMode ( ) check before adding URL to auto suggestions list and saving history
Allow zooming on map when base zoom is within limits
Added methods isFileUnderVcs ( Project , VirtualFile ) and isFileUnderVcs ( Project , FilePath ) to VcsUtil class .
Update copyright year in FileTemplateTab.java and import JBColor for modified foreground color .
Remove unnecessary import statement for org.jboss.netty.handler.codec.http.HttpResponse.
Remove redundant boolean check in MethodDescription.java
Fix illegal argument exception for maxContentLength &lt; = 0
Added import statement for MessagingException
Remove unnecessary code in NioEndpoint
Update Cassandra Storage class to use new Thrift API
Fix race condition with binding in EmbeddedApplicationSupport
Added -XDdev to DEFAULT _ ARGS in CompilationTestHelper.java to print stack traces for completion failures .
Added note about Referer header in WebView.java
Fixed endpoint mapping for ManagementWebSecurityAutoConfiguration
Removed unnecessary metric fetching for Graylog2
UnmanagedThreadException now extends ExecutionException instead of RuntimeException
Added render pass tracking and hooks for Forge client .
Removed unused import statement for FileSystemSubset in GroovyRuntime.java.
Added import of NoSuchEntityException and updated exception handling in EjbExceptionTransformingInterceptorFactory .
Added import statement for ExtensionPopupMenuComponent and updated isEnable and isEnableForInvoker methods to use it .
Added flag parsing to CommandContext
Added a check to ensure that the semaphore is released even if client.bulk ( ) fails .
Added version label to DesignerEditorPanel and PaletteItemsComponent .
Renamed import from XmlFile to PsiFile and updated method signature to use PsiFile instead of XmlFile
Update BoundaryTraceIdSampler.java to use a constant expression for readability .
Added loadFragmentIndex method and updated onCheckedChanged to use it
Added co-author Gerard Davison to DeclarativeLinkingFeature.java
Fixed issue withFocusCommand.saveAllocation ( ) not properly initializing myAllocation in debug mode .
Fix issue with package name retrieval in ActivityManagerService
Added check for null Ipo before creating ObjectAnimationModifier
Added buffer limit check in FileBuffer.write ( ) method to prevent exceeding buffer size .
Remove activity monitor in finalize method .
Added patch version 4 to HandlerFactory.java.
ZAP : Hook OverrideMessageProxyListener into the Proxy
Added &apos; ne &apos; and &apos; sheq &apos; methods to IR class .
Added support for standard group by clauses in SQL parser .
Fixed illegal state exception message in StandardSession.java
Added CompoundDrawables to ShadowTextView .
Added debug logging for flushed bytes in ChannelEndPoint
Added import statement for Collections and modified getStates method to return an empty list instead of a single element list when the state is null .
Update copyright year in SingleConnectionFactory.java from 2014 to 2015 and add a temporary copy of delegates to avoid ConcurrentModificationException when iterating over listeners .
Added Content-Type header to SimpleErrorPageHandler
Reinit primary method updated to allocate lights and matrices only once .
Set AboutWindow to be non-resizable
Added initialization of TaskRabbit in WebServicesView
Fixed issue with resource copier by removing unnecessary copyScript.invokeMethod ( &quot; run &quot; , new Object &#91; 0 &#93; ) call .
Dispose root frame when quitting application
Fixed MalformedURLException handling in RequestBuilderBase
Skip checks for value types when only modifiable fields are present .
Added log statement for error handling in LocalMUCUser.java
Added support for Glass compatibility and improved battery / CPU performance by adding a minimum preview FPS range .
Added &apos; type &apos; field to NetworkEventReporterImpl.params and updated peerManager.sendNotificationToPeers ( ) calls to include the new field .
Modified VerifySegmentState.java to updated table prefix option usage .
Added ability to disable upload timeout in AbstractHttp11Processor
Removed unused variable &apos; verifyHost &apos; in SSLHelper .
ZapNumberSpinner : Changed constructor arguments order to ( minValue , defaultValue , maxValue ) from ( defaultValue , minValue , maxValue ) . ZapPortNumberSpinner : Changed constructor argument order to ( MIN _ PORT , defaultPortNumber , MAX _ PORT ) from ( defaultPortNumber , MIN _ PORT , MAX _ PORT ) .
Removed unused import statement and simplified peer accessor checks in PeerGroupTest .
Added disable _ gl _ preload property and disabled OpenGL preloading by default .
Increased wait time for response from 15 seconds to 30 seconds in AsyncTwitterTest .
Added documentation comments to DBManager and App classes
Remove unnecessary System.out.println statements in TaskManager and RuntimeTask .
Fixed nullity check for fields with regexp validation
Added version checking to isValidSdkHome method in GoSdkType .
Removed unnecessary &apos; ourPrimitiveTypeMapping &apos; field from EvaluatorBuilderImpl .
Fixed issue with incorrect calculation of extra width in BaseTableLayout.java
Added setTitleVisibility ( View.VISIBLE ) in JCVideoPlayer to show title when video is playing .
Added import statement for gherkin.formatter.model.Comment , and changed the construction of Step objects to use Collections.emptyList ( ) instead of an empty array list . Also fixed a comment .
Added check for null listener before invoking callback method in FileDownloadEventPool
SvnVcs : fixed issue with change lists synchronization
Added method isWorkerRegistered to BlockMasterPrivateAccess to check if a worker is registered with the block master .
Fixed issue with ClassData retrieval in PackageAnnotator
Added import statement for buildcraft.core.BlockBuildCraft and modified BlockTank class to extend BlockBuildCraft instead of BlockContainer .
Added check for current amount of fluid before draining
Added synchronized code to ensure that commits are properly executed in TxMaker .
Fix compilation errors and improve error handling
Added new JavaScript examples : Environment , HTML5 , Libraries , Services
DiscretizedStream instead of WindowedDataStream for mapWindow
Update to use Jimfs with Unix configuration for in-memory file system .
Added support for SASL EXTERNAL authentication in ConnectionConfiguration .
Added support for loading Blender models with packed and referenced images .
Added warning for conditional calls to id generators .
Added new method getResultsStore and called it in run method . Also , added a new BenchmarkResultsPrinter class .
Removed unnecessary @ SuppressWarnings annotations in TestStandardContext and TomcatBaseTest .
Fix concurrent modification exception message to correctly attribute the cause of the conflict .
Added check for SD card readiness before exiting FileViewActivity
Fixed printing of CompositeExceptions to include all wrapped exceptions .
Fixed exception handling in NioEndpoint
Fixed getValueAsString method return type from Object to String in VariantsMetaData interface .
Set node health value to zero when index count is zero
Deprecate @ EntityScan in favor of explicit configuration or @ org.springframework.boot.autoconfigure.domain.EntityScan
Fixed password reset for isolated contexts in DataSourceConnectHandler .
Modified TextureRegionLibrary constructor to take an integer parameter for initial capacity and removed unnecessary import statement .
Added option to expose keeping punctuation in ParserAnnotator and updated verbose output for loading parser model .
Added support for adding multiple filters when receiving a filter category .
Added temperature sensors to privacy restrictions .
Renamed DataServerIntegrationTest to DataServerTest
Added import statement for SingletonFileTree and updated visitor pattern to use SingletonFileTree instead of DefaultFileVisitDetails
LeaderInquireClient : Log error message instead of just the error message
Fixed issue with ListPopupImpl &apos;s getPreferredScrollableViewportSize ( ) method . Now correctly calculates size based on visible row count and model size .
Added error message for JSON processing exceptions
Removed unnecessary System.out.println statement in IOConverterTest.testInputStreamToString ( ) method .
Added support for additional browsers in Util.java.
Extended TenantJson class with additional properties and constructor
Fixed a bug in RunManagerImpl where the unique name for run configurations was not being generated correctly for unknown run configurations .
Added EmptyTransportResponseHandler.INSTANCE _ SAME to sendJoinRequestBlocking method
Added visible and opacity attributes to TmxMapLoader .
Added check for zero-sized characters in Font.updateTextBounds ( ) , and threw IllegalArgumentException if character has no extent .
Added check for empty databaseChangeLog in DiffResult.java
Added guard against rare thread races in TaskManagerTest
Added a commented out test for assertIndicesDirsDeleted ( nodes ) in IndexWithShadowReplicasIT.java.
Added Apache License header to XpathIcons.java and updated icon sizes .
Added mappings for C + + header files ( .hh and .hpp ) and updated file type identifiers for C and C + + source files
Added resizable and movable properties to ShowImplementationsAction popup
Added test for merging socket servers without ports and fixed mergeServer method to return ActualSocketServer instead of SocketServer
Added fused location provider to LocationManagerService
Fixed rare issue with recovery target service where local mapping changes were not being properly accounted for during translog replay .
Added code to set title of toolbar button to active project file
Fix Java runtime UTF-8 support error
Increase worker thread pool size for cluster node and client
Added NonProjectScopeDisablerEP to disable search in non-project items
Added logging for version resetting in Gossiper and fixed version resetting in IncomingTcpConnection and OutboundTcpConnection .
Fixed bug in AtmosphereRequestImpl.java - correctly set HttpSession for request now .
Update PlayApplicationPlugin to use buildDir instead of temporary Dir for analysis file
Added import statement for java.lang.reflect.Modifier ; and updated Utils class to skip static fields when building JSON representation of an object .
Added table writer count setting for local execution planner
Added namespace awareness to XmlTestResultParser .
Added configuration field to InputsResource
Added support for dash notation in JSON Path
Modified PetValidator to add additional validation for pet name and type , and to require a birth date for new pets .
Updated link to Little &apos;s Law in RateLimiter.java
Adding fare composite validation
SEC-1188 : Add getContextHolderStrategy ( ) method to retrieve context strategy
Added convenience GET method for adding events through query string
Updating ExponentialBackoffMsgRetryManagerTest to use longer delays
Fixed issue with selection handles in WebViewClassic.java
Added check for non-native table before applying aliasing for TableScanOperator
Added spring-context-support dependency to GrailsCoreDependencies .
Fixed invalid tag attribute value parsing in FontListParser.java
Added TachyonFS and FileInStream imports to FileInStreamIntegrationTest .
Renamed &apos; Packet &apos; class to &apos; Stanza &apos; and updated documentation to reflect new name and added explanation of XMPP stanzas .
Added support for WHPP in UniversalSemanticHeadFinder .
Removed Spark Help button from ConnectionsPane
Fixed issue with Validator.isSkipped ( ) method .
Added code to fireProviderPresenceStatusChangeEvent and initialized ssContactList in OperationSetPersistentPresenceJabberImpl .
Renamed event bus log tag from &apos; Event &apos; to &apos; EventBus &apos; .
Added new parameters for elastic averaging in DeepLearningParameters
Stopped sounds for incoming and outgoing calls .
Added delayed screenshot logging in BugreportProgressService .
Added color filter for favorite image drawable
Remove Gradle Java Home property setting in GradleImportingTestCase .
Added two new test cases : testHavingWithoutGroupBy and testGroupByAsJoinProbe to AbstractTestQueries.java
Update SingleResultSet.java to use Collections.EMPTY _ SET instead of HashSet for empty result sets .
Added check for zero notifications before canceling them .
Modified engineUpdateAAD method in AndroidKeyStoreCipherSpiBase.java
Added support for XML Language in BraceMatchingUtil
Added @ SuppressWarnings annotation to JsonStoreBuilder.build ( ) and added a new constructor to JsonObjectIterator that takes a StoreDefinition as a parameter .
Added timed transfer edge and updated transfer table to handle specificity .
Removed unnecessary &apos; adjustSize &apos; variable in add method .
Fixed exception with null VirtualFile in GitChangeProvider.unversioned
Update JsonReader and JsonWriter classes .
Added visitor pattern for parse trees .
Removed unused method setChainHeight ( int ) from WalletProtobufSerializer .
Updated ContactsContract class with new content : / / style URL for looking up data rows by email address and updated filter API .
Fix leaky classloaders in DefaultIsolatedAntBuilder
Fixed issue with LookupImpl.fillInCommonPrefix ( ) not setting focused state correctly when explicitly invoked .
Added import statement for PathEffect and removed unnecessary import statement for Collections .
Increased timeout for pingPongTest from 90000 to 180000
Removed unresolved contacts from ContactGroupIcqImpl
Change uid from negative to positive in CSetting constructor .
Added unwrapped ( ) method to ConfigObject interface and implemented it in AbstractConfigObject
Removed unnecessary log statements in CordovaWebView.java
Refactored getEpisodesAndUpdateDatabase to use void return type and removed unnecessary boolean variable .
Added blocking IO for reloadable mode in ReloadableFileBackedFactory
Hide fullscreen button in JCVideoPlayerSimple
Added opaque property to myTfFullPath text field in SourcePathsStep.java
Fixed issue with environment variables parsing
Added onSelection event handler to ChooseFolderDialog2 to set the selected folder path in the filename text field .
Added synchronized method close ( long socket ) to AprEndpoint to prevent JVM crash when calling it more than once per socket .
Ignore duration notation with &apos; P &apos; in FlagEncoder
Added support for mutating background drawable in SubActionButton .
Remove unnecessary check for idea.properties in PathManager .
Removed unnecessary imports : org.greenrobot.greendaotest.DaoSession , org.greenrobot.greendaotest.SimpleEntityDao
Added &apos; final &apos; modifier to &apos; uri &apos; and &apos; errorMessage &apos; fields in DeepLinkError class .
Added &apos; break &apos; statement to &apos; CDMAPhone &apos; class to handle unexpected events .
Added capability CAP _ BLOCK _ SUSPEND to startSystemServer method
Fixing configuration exception for sense service
MongoCredentials.java : Hidden password in toString ( ) method .
Added importScope for class header resolution and member declaration resolution
Added logging for command execution in CommandRuntime .
Replace backslashes with forward slashes in absolute resource path .
Added code to take screenshots of activities when they are finishing , and updated thumbnail for hidden activities .
Allow IcedHashMapBase to be deserialized with an empty map
Added import statement for &apos; org.apache.camel.spi.DataFormat &apos; and updated import statements for &apos; org.apache.camel.support.ServiceSupport &apos; in AbstractXmlDataFormat.java.
Removed unreachable code in InstrumentMethod
Added immutable naming strategy with suffix and Java Lang package prefix
Added support for creating child directories for library projects .
Added explicit type casting for value at index 1 in TestSubscriberTest
Update media player to select default subtitle track on metadata update .
Fixed incorrect log message in ImageUploadTaskService.java
Updated regular expressions for parsing docstrings .
Added ability to add multiple libraries at once and updated adapter notifications
Added hashCode method and modified equals method in IPAddress class .
Modified BufferUtils.java to change the logging message when failing to unmap a direct ByteBuffer .
Fixed bug in scheduleDrainBuffers method of BoundedLocalCache class
Added JMX export support for Endpoints
Eviction policy evaluator : Implemented LFU eviction policy .
Removed unused &apos; KillProcessAction &apos; import and removed unnecessary action from the toolbar .
Added null check for boxed arguments in Reflector.boxArg method .
Added onChildVisibilityChanged method to GridLayout to invalidate structure when a child &apos;s visibility changes
Added import statement for JsMessenger and modified convert method in JsContact to use JsMessenger instance .
HouseKeeper : Changed System.currentTimeMillis ( ) to clockSource.currentTime ( )
Added conditional bean annotation to YubiKeyConfiguration class .
Added import for EncodingProjectManager and set default charset name to UTF-8 in ConvertToBasicLatinTest .
Added dispatchTouchEvent call in DragSortListView to handle touch events properly .
Remove unnecessary comment in JavaSymbolFinderIntegrationTest.java
Removed unused variable &apos; r _ version _ home &apos; from ActiveSession.java
Added check for null before splitting locations in PropertiesComponent # setLocation
Modified DirectionDrawable.java to use canvas.getHeight ( ) / 2 and canvas.getWidth ( ) / 2 instead of width / 2 and height / 2 for rotation anchor point .
Update copyright year in AbstractRefreshableConfigApplicationContext.java and replace system property values with environment property values in resolvePath method .
Update theme settings to read from &apos; theme / theme.txt &apos; instead of &apos; theme / settings.txt &apos; .
Removed unused constructor from PutAllBackupOperation and PutAllOperation .
Removed unnecessary System.out.println statement in TestParser.java.
Added check for non-empty collection of attributes before breaking out of loop in DelegatingMethodSecurityMetadataSource # getAttributes ( Method , Class )
Removed unnecessary @ Before method in ActiveDirectoryAuthenticationIT test class .
Added full-screen width support for notification panel .
Added method &apos; flushStatements ( ) &apos; to interface SqlSession .
Update to WebKit codebase to version 96bf46e . Changes include : &#91; diff --git a / core / java / android / webkit / CallbackProxy.java b / core / java / android / webkit / CallbackProxy.java \ nindex b2277cb .. 96bf46e 100644 -- a / core / java / android / webkit / CallbackProxy.java \ n + + + b / core / java / android / webkit / CallbackProxy.java \ n @ @ -111,7 + 111,7 @ @ class CallbackProxy extends Handler { \ n \ t / / Result transportation object for returning results across thread \ n \ t / / boundaries . \ n-private class ResultTransport &lt; E &gt; { \ n + private static class ResultTransport &lt; E &gt; { \ n \ t / / Private result object \ n private E mResult ; \ n diff --git a / core / java / android / webkit / WebIconDatabase.java b / core / java / android / webkit / WebIconDatabase.java \ nindex d284f5e .. 6cc6bb4 100644 -- a / core / java / android / webkit / WebIconDatabase.java \ n + + + b / core / java / android / webkit / WebIconDatabase.java \ n @ @ -37,7 + 37,7 @ @ public final class WebIconDatabase { \ n private final EventHandler mEventHandler = new EventHandler ( ) ; \ n / / Class to handle messages before WebCore is ready \ n-private class EventHandler extends Handler { \ n + private static class EventHandler extends Handler { \ n \ t / / Message ids \ n static final int OPEN = 0 ; \ n static final int CLOSE = 1 ; \ n diff --git a / core / java / android / webkit / WebStorage.java b / core / java / android / webkit / WebStorage.java \ nindex c3b359e .. ae560fb 100644 -- a / core / java / android / webkit / WebStorage.java \ n + + + b / core / java / android / webkit / WebStorage.java \ n @ @ -70,7 + 70,7 @ @ public final class WebStorage { \ n private Handler mHandler = null ; \ n / / Class to handle messages before WebCore is ready \ n-private class Origin { \ n + private static class Origin { \ n \ tString mOrigin = null ; \ n long mQuota = 0 ; \ n
Added method getWhenElseBranch to JetPsiUtil , and updated skipTrailingWhitespacesAndComments to use PsiTreeUtil.skipSiblingsForward
Added SplashOutputWriter to root logger
Added support for assignment expressions and logical operators in Groovy PSI creator .
Modified NetworkImageView to use setDefaultImageOrNull ( ) instead of setImageBitmap ( null ) for better performance .
Added logic to handle selection of text ending with a dot or space character .
Collapse cursor to beginning or end of selection in FindReplace ,
Fixed JetAnnotationEntry.java to use getParentByStub ( ) instead of getStubOrPsiParent ( ) for getting the parent element .
Modified log level from DBG to VDBG in makeWimaxStateTracker ( ) method .
Added check for task completion in concurrent engine usage test
Added import for ElasticSearchParseException and added check for START _ OBJECT token in XContentSettingsLoader
Implemented logic for handling multiple lambdas in GLM2 .
Replace SSL connector with non-SSL connector in SampleTomcatTwoConnectorsApplication .
Added test for loading script from cache with stale cache
Added FileUtil.toSystemIndependentName ( path ) to correctly handle paths on different operating systems .
Removed unused OnDismissListener implementation in CrashReportDialog .
Update CommonActionsPanel.java to change the condition for decorating buttons from SystemInfo.isMac to UIUtil.isUnderAquaLookAndFeel ( ) for better compatibility with Aqua Look and Feel .
Added update flag for shows last updated more than 12 hours ago
Remove unnecessary cast in Array.tabulate
Added commitAllowingStateLoss ( ) to FirstUsageWelcomeFragment and FirstUsageWizardFragment
Added Apache HTTP client connection pooling to RestClient
Added &apos; myApplied &apos; field and updated &apos; markApplied &apos; method in NoConflictChange class .
Added import statements for GrNamedArgument and updated code to use GrNamedArgument in GroovyCompletionContributor .
Added support for repeated RAPTOR searches in profile routing using PropagatedTimesStore .
Added check for server shutdown in SessionManager.getSession ( ) method .
Removed import statement for de.hunsicker.jalopy in app / Editor.java
Added try-catch block for camera startPreview and internalShoot methods to handle exceptions and added cancelAutoFocus method .
Added import of Java Security library for AES encryption and updated method signature to throw NoSuchAlgorithmException .
Added Apache License 2.0 header to DefaultAsyncSupportResolverTest.java
Added support for multiple headers with the same name in NetworkEventReporterImpl .
Added support for tox file in different directory
Added dotall flag to SSID _ PATTERN to allow scanning of hidden SSIDs .
Added check for gocheck import path in GocheckFramework
Added import statement for java.io.Serializable and implemented Serializable interface in MyBean class .
Renamed GroovyJava7RuntimeMemoryLeakStrategy.java and MemoryLeakPrevention.java to use public access modifiers .
Update URI creation for JavaClassFromFile constructor
Fixed bug in ConnectorManager stop method
Update logging level for translog file not found from info to debug
Added method getNaturalEndpoints to StorageServiceMBean and NodeProbe to retrieve endpoints responsible for storing specified key .
EditorMarkupModelImpl : adjusted calculation of preferred size to account for vertical scrollbar width
Fixed setReadOnly method in HiveConnection class to throw a SQLException if the connection is closed or if the request tries to enable read-only mode .
Added check for loaded font and threw RuntimeException if not loaded
Added self-type parameter to FunctionType when inferring type
Added logger and output directory to BundleExtractCommand
Added optional wallet parameter to BlockChain constructor and added a new constructor that creates a BlockChain without a wallet .
Modified DefaultUrlMappingInfo.java to change the way action names are evaluated .
Added method &apos; order &apos; to HibernateCriteriaBuilder class
Added try-catch block for programmatic installation of Byte Buddy agent to handle exception and improved error message .
Added parameter listener to TwitterStreamFactory constructor
Added WPStats import and tracking for stats accessed
Added version information to AtDXmlSerializer.java
Fixed a bug where a duplicate broadcaster was created when using the DefaultBroadcasterFactory .
Added null check for address parameter in placeCall method .
Added video link and improved disjoint set implementation using path compression and union by rank .
Removed unused &apos; recycle ( ) &apos; method call in Http11Processor .
Added cancel ( ) method call to JobHandle in ContinuousJob.java to properly cancel the job when stopping
Added VcsDirtyScopeManager import and updated PatchApplier to use it .
Added import statement for &apos; com.intellij.openapi.util.Disposer &apos; and added call to &apos; Disposer.dispose ( myTreeBuilder ) &apos; in &apos; disposeUIResources ( ) &apos; method .
Added support for concurrent access to lifeCyclesCalled map in TestEncodingDecoding.java
Update greeting method call from &apos; org.jetbrains.namespace.getGreeting ( ) &apos; to &apos; org.jetbrains.JetbrainsPackage.getGreeting ( ) &apos; in HelloWorldJavaTest.java.
Fixed potential stack bug in OperationSetBasicTelephonySipImpl by ignoring unnecessary server transaction check .
Added override for toString ( ) method in AbstractPatternRule and PatternRule classes .
Added webref and improved formatting for XML data .
Removed unnecessary rebuilding of state map for all blocks .
Disable creation of launcher script in InitialConfigurationDialog
Removed unnecessary &apos; readBuffer.limit ( nRead ) &apos; line in NioServletInputStream.java
PathManager.java : Updated getAndroidEmulatorRoot ( ) method to use getAndroidSdkRoot ( ) instead of hardcoded path .
Renamed variable &apos; bone &apos; to &apos; boneName &apos; in BoneContext.java
Disable Mac OS X specific code for Java 1.5 due to a Java bug .
Fixed non-deterministic coin selection in WalletTest .
Fix bug in VariableContext : handle shadowing of variable names correctly
Added emphasis to greetings using HTML tags .
Fixed case sensitivity issue in Roles.java by removing .toLowerCase ( ) method call
Renamed method from executeStart to renderEnd in PrettyPrinter class .
Added support for transaction isolation level in PoolBase
UpdateGenerator : Properly handle Java util Date and Boolean values
Renamed &apos; CommonRefactoringUtil &apos; to &apos; getTitle ( ) &apos; and updated the error message accordingly .
Added import of Apg library for crypto app and set mCryptoApp to Apg.NAME.
Added documentation for enabling protected mode in JNA
Set alpha of background to 255 in SweetAlertDialog
Fixed issue with header handling in AtmosphereResponse.java
changed Hashtable to HashMap in CreateFont class
Added error type code to message format string .
Replace &apos; INSTANCE &apos; with &apos; final INSTANCE &apos; in VirtualMachineMetrics.java
Added conditional logic for adding shortcuts to GlobalShortcutDialog
Added method &apos; shouldPassSneakingClickToBlock &apos; to ItemWrench to allow sneaking click to pass through to the block .
Update RepositoryHandler to use system property &apos; maven.repo.local &apos; instead of &apos; maven.repo.path &apos; for determining repository location .
EpisodeDetailsActivity : removed unnecessary ActionBar initialization and added setupActionBar method to handle ActionBar settings .
Changed parameter name from &apos; index &apos; to &apos; composingTextStart &apos; in method setComposingText of class CursorAnchorInfo .
Fix for dismissing action mode popup on pre-v18 devices
Removed unnecessary code in CodeFormatterFacade.java
Added deviance function for GLMModel
Updated MultipleChangeListBrowser to use ActionPlaces.TOOLBAR instead of ActionPlaces.UNKNOWN for creating action toolbar .
Added hide preferred network type setting and updated default values for carrier config
Added methods for accessing values of specified fields and elements in TreeNode interface .
Added test for ClassCastException in Java serialization and Pojo serialization with writeReplace ( ) method .
Added import of org.apache.camel.model.ModelCamelContext and updated context variable from CamelContext to ModelCamelContext in AbstractCamelRunner .
Regenerate background instead of invalidating when setting selection drawable
Removed unnecessary import statements in ParkingPluginActivity.java
Added test case for relative location URL redirect
Added import statement for NetworkUtils and updated PlansActivity to check for network connection before downloading plans .
Added builder factory and build functions to Type interface and BasicType class .
Removed unnecessary focus listener from MemberSelectionPanel .
Removed nativeDrawGL function in WebView.java
Removed unnecessary import statement for MultiDex and updated attachBaseContext method to use MultiDex.install ( this ) ;
Added cache monitoring and metrics support to CliHistorical .
Replace ThreadContextObjectInputStream with CustomObjectInputStream in SerializedCache
Added @ Deprecated annotation to WebMergedContextConfiguration constructor .
Update ExplodedArchiveTests to use File.separator instead of hardcoded path separators .
Added support for non-physical file systems in PsiManagerImpl
Removed Lombok @ Getter annotation and added getId ( ) method to Edge class .
MavenUIUtil : updated background color logic for selected items
Added thread pooling to HttpClient
Added homeAsUp and title string in Example _ ActivityBaseClass .
Removed unused import : org.elasticsearch.threadpool.ThreadPool
Added setQualifier method to JsInvocation class .
Removed unnecessary System.exit ( 1 ) call in Session.java
Added import statements for io.undertow.server packages
Added import statement for TextAttributes and added a new HighlightInfo creation method with improved error messages and added null checks for highlightInfo and quickFixActions .
Added error handling for AssertionError in ChangeListManagerImpl .
Fixed illegal argument exception for unknown media types in CallPeerMediaHandler .
Added new visibility options for notifications : VISIBILITY _ PUBLIC , VISIBILITY _ PRIVATE , and VISIBILITY _ SECRET .
Added test for getting type of abstract factory bean
Added MapMaker to create soft value ConcurrentMap in ScriptService
Added quick fix for unresolved references in Python decorators
Added intent to start SettingsActivity from DetailActivity and MainActivity
Added alarm to delay showing &apos; too much output &apos; notification
Added support for WebSocket , Long-Polling and Streaming in AtmosphereHandlerPubSub and MeteorPubSub samples .
Remove unnecessary save ( ) call in detectLanguageAndEncoding ( ) method of NativeFormatPlugin .
Added test for glob in implicit outputs
Removed unnecessary android.R.id.home item from SitePickerActivity &apos;s menu and made getAdapter ( ) private .
Added long clickability to RecipientSelectView .
Updated TSVSentenceIteratorTest to use BASIC dependencies instead of STANFORD dependencies .
Fixed SSLHandshakeException handling in HostnameVerificationTest and SslConnection .
Added setLogInterceptor method to HawkBuilder and removed interceptLog method .
Update AlluxioFramework.java to set Mesos master user to the current user .
Fixed grammar mistake in Context.java
Added logging for handling read events when not heartbeating
Added ENABLE _ TAGGING flag and updated internalModuleCtrl method to respect tagging disabled setting .
Added &apos; protected &apos; access modifier to method disambiguateUntil in DisambiguationRuleTest.java.
Added import statement for List and updated OOrientDBLoader class to use ODatabaseRecordThreadLocal instead of OBasicCommandContext
commit message here
Fixed bug in Compiler.java : analyze ( ) method now correctly handles context switching for EVAL and STATEMENT forms .
Removed unnecessary code in AndroidDomExtender
Fixed invalid type name exception in MetaDataMappingService and removed unnecessary refresh call in IndexShard postRecovery .
Removed unnecessary call to setVisible ( true ) in LwjglFrame constructor .
Added support for DRM keys loading notifications .
Modified printStart to use generation unit source name instead of output path .
Added system API constant for query package restart and extra for packages
Added @ BeforeClass annotation to DateIndexTest.java
Added support for LDAP v3 compatible servers by introducing a new &apos; useLdapContext &apos; flag and updated the &apos; InitialDirContextFactory &apos; class to use &apos; InitialLdapContext &apos; for LDAP v3 servers .
Fixed bug in UniqueConstraint.java where null table names were not being compared correctly .
Added empty string to label in ChatRoomConfigurationWindow
Fix for J2ObjcClassInfo generation when type &apos;s package is null
Changed QueryStringJsonQueryParser to use false for escape instead of true .
Fixed issue with null pointer exception in VcsLogImpl.getDetailsAtRow ( ) method .
Added methods for obtaining specialized views of keys and values in ImmutableOpenMap
Update FileSystemMasterIntegrationTest to set / remove AuthorizedClientUser correctly .
Added logging for IOException in GitBranch constructor
Modified RecentsPanelView.java and WindowManagerService.java files .
Added final modifier to GitHubService field in NetworkApi , and replaced PublishSubject with BehaviorSubject in RepositoriesViewModel .
Added method clear ( ) to class Transcript to clear the events list .
Fixed issue with ResultSetPersister.java , now uses model.getCellValue instead of row.values &#91; binding.getOrdinalPosition ( ) &#93;
Added import statement for org.jetbrains.annotations.NonNls ; and changed getVirtualFile method to use @ NonNls String filePath parameter .
Added imports for concurrent utilities and added a shutdown hook for Orient .
Removed unnecessary author line in Bootstrap.java.
Update TextureManager to reload textures that need to be updated on hardware .
Fix error message for OIndexUnique
Added import statement for BuildManager and updated GroovyCompilerConfigurable class to use it .
Added saveCurrentSelections ( ) call in clearAllGpxFileToShow ( ) method .
Added import statement for &apos; InitBinderDataBinderFactory &apos; and replaced &apos; ServletRequestDataBinderFactory &apos; with &apos; InitBinderDataBinderFactory &apos; in method signature .
Added custom _ layout6 , stateFoo , and MyBlueTheme to R.java
Removed unused variable &apos; radian &apos; from Instruction.java and RoundaboutInstruction.java
Added constructor to Printer class
Added accessibility support for SearchAddressFragment &apos;s menu items .
Added import statements for water.H2O and overridden methods for JSON serialization / deserialization in Enum class .
Added CompilationParam.CHECK _ LINT and CompilationParam.CHECK _ SYMBOLS
Fixed night mode text color and shadow color for MapInfoLayer .
Removed deprecated setSSLContext method and added a warning to use SslContextFactory instead .
Update ServerSystemTest to include additional assertions for Elasticsearch and Compute Engine
Increased file size limit from 500 KB to 5 MB and updated large file size calculation for Desktop environment .
Replace NeoStore with NeoStores in IndexSamplingIntegrationTest .
Added check for route calculation before restoring routing mode .
Added catch blocks for TimeoutException , IOException , SmackException , and XMPPException in Socks5BytestreamRequest.java
Fixed issue with space in application path and system-handled applications .
Change type of variable &apos; pair.type &apos; from &apos; requiredType &apos; to &apos; JSType.UNKNOWN &apos; in NewTypeInference.java and add a new test case for computed access to a useful type in NewTypeInferenceES5OrLowerTest.java.
Modified SubsampleTests.java to change output shape from ( numExamples , 1 , 19 , 19 ) to ( numExamples , 1 , 10 , 21 )
Update AppendProcessor to use compiled templates
Added lazy unmarshalling for OMVRBTreeRID
Fixed incorrect implementation of ChannelInboundHandler and ChannelOutboundHandler in ChannelPipeline.java.
Added ability to recalculate Bloom filters for peers and returned a ListenableFuture for the operation .
Added @ Deprecated endpoint method to ServiceRegistration
Decorate a day by making the text big and bold instead of underlining it
Added loop to iterate over child registrations in ProxyControllerRegistration
Fix for SequenceActor &apos;s isCausesInvalidation method to correctly handle FatSeqUpdates
Added support for offline resources in HelixInstanceDataManager by updating the resource name passed to addResourceDataManager .
Fixed issue with expanded nodes not being properly restored in XDebuggerTreeRestorer .
Added methods getRenderer and getAlgorithm to ClusterManager class and modified PolyUtil class .
Removed unused import org.apache.catalina.startup.HostConfig ; in StandardHost.java and updated TestRegistration.java to expect 20 Tomcat MBeans instead of 21 .
Added thumbnail column to Browser class
Added method setImage ( Image bgImage ) to reset the background image for this button .
Added support for getting tokens by index in UnbufferedTokenStream .
Allow subsequent calls to setLocation to override any previous call in HelpPane .
Added flexbox grow properties to ChunkDataWidget
Added dependency parser neural network package .
Update test documentation for embedded examples and kernel tests
Fix up locale list in Configuration.java
Added a check to break out of the loop when the length of the other token array is reached , to prevent an infinite loop in the case of a wildcard in the thisToken array .
Fixed bug in MethodValidator where methods with non-void return types and non-empty parameter lists were not properly validated .
Added display distance direction and type string for amenities in AmenityMenuController .
Added android.tests.dependencies to EXCLUDED _ FILES _ AND _ DIRS in JetCodeConformanceTest.java
Update FileContentType to use DEFAULT _ CONTENT _ TYPE _ WITH _ CHARSET instead of DEFAULT _ CONTENT _ TYPE
Added project instance to GradleBuildProcessParametersProvider and updated classpath calculation to use ExternalSystemApiUtil .
Fixed issue with incorrect TLD URI suggestions in JSP files
Update MetricExportProperties.java
Removed unused import org.gradle.model.internal.type.ModelType
Fixed bold font for non-default buttons on non-Mac platforms .
Remove unused imports and fields in OperatorStats
Modified TextBodyBuilder.java to add setter methods for includeQuotedText , insertSeparator , signatureBeforeQuotedText , replyAfterQuote , and appendSignature .
Added bind hook to CocoaMessenger &apos;s message display list to load more history on scrolled to end and handle item touched events .
Added support for downloading files from HTTPS URLs .
Animations : Adjusted duration of show animation from 667 to 500 and added listener to ScaleUpAnimHelper .
Added parameters --branches , --remotes , and --tags to GitHistoryUtils.java.
Remove unused import org.picocontainer.Startable in DefaultDatabase.java
Stop debug process when exited
Added primary key constraint to lineitem table and updated column names
Ignoring roots changed events caused by dumbness change
Modified InBandBytestreamSession.java to return byte and increment buffer pointer using bitwise AND operator instead of cast .
Added getFile method to return the model file and updated saveModel method to use SerializationUtils.saveObject
InstrumentedThreadFactoryTest : increased timeout for awaitTermination from 1 to 2 seconds .
Replace LibC chmod implementation with EmptyChmod on LinkageError
Fixed mistake in PatternTestTools.java : removed unnecessary single char disjunctions
Added retry logic inspired by droid-fu project
Added endAccess ( ) call in PersistentValve.java to properly clean up session resources .
Modified CodegenTestsOnAndroidGenerator.java : removed unused variable &apos; counter &apos; and simplified code .
Added scroll flag check for minimum offset calculation in SmoothAppBarLayout
Updated JsonStringToMap.java to use json schema instead of chararray
Added DEBUGGABLE _ ERROR message to AndroidOSProcessHandler
Added file format version attribute to StorageExporter.java
Removed unnecessary scheduleAsyncPersist method in LineageFileOutStream
Update VectorDrawable.java to clarify that clip paths only apply to the current group and its children .
Fixed bug in GoCompletionUtil.createVariableLikeLookupElement : now returns null if variable name is empty
Update JavaScriptProtocol to use milliseconds for heartbeat interval
Fixed issue with CommitChangeListDialog not showing correct comment for alien changes
Added call to changeStatus ( Account.State.OFFLINE ) when closing TCP connection without closing stream .
Fixed issue with cell value retrieval for nested attributes
Added check for missing closing brace in PdePreprocessor
Fixed typo in MemberStateImpl.java
Fixed incorrect standalone flag in StructDatabase and FieldTree .
Added wakelock to RemoteImService to prevent device from sleeping during network state changes .
Fix typo in AbstractTask.java ( leftShit - &gt; leftShift )
Added import for android.text.format.DateUtils ; and added a check for daemon connection before running fstrim idle maintenance
Added field &apos; executorType &apos; and &apos; statementType &apos; to MyBatisEndpoint.java
Updated SignInFragment to use the correct title for the application log
Bas Leijdekkers updated copyright header in UnnecessaryAnnotationParenthesesInspection.java.
Added Hopper block with centralTopLimit of 0.625
Update BaseDexExceptionHandler to extend RuntimeException instead of Exception
Added animation to UserPhotosFragment
Added -Djava.io.tmpdir option to GradleDistributionExecuter
Disable unnecessary environment variable settings for Genrule .
Added method getRequestType ( ) to ONetworkProtocolBinary and updated OClientConnectionManager to skip shutdown requests .
Increment cache version and add android.support.v4.util.LruCache to instrumenting exclude list .
Added test for replacable lookup in AcegiMessageSourceTests
Removed unused imports and variables , simplified code
Added refcount and getRefcount methods to Identifier class .
Added conversion of location object in onLocationChanged method for proper handling of location updates .
Added import statement for PathManager and modified getSettings method to include PathManager.getConfigPath ( ) in the list of directories to search for EditorConfig files .
Added methods getGenJar and getGensrcJar to AndroidCommon class .
Update copyright year in ConcurrentSoftValueHashMap.java from 2014 to 2015 .
Fixed issue with SctpMessage construction , now correctly duplicating content buffer instead of copying it .
Added UppercaseSentenceStartRule and removed CommaWhitespaceRule in Persian.java
Added assertion for BridgeAssetManager in BridgeContext constructor
Removed unnecessary variable assignments and added debug statements for better logging .
Update PlainSaslTransportPlugin to use new CallbackHandler implementation
1
Added SpiceProfilingUtil log and profile calls for retweet action
Added method toString ( ) to PyDocstringParserDefinition class .
Update ChannelHandlerContext.java to change context type from class to interface .
Update natural language processing logic relations .
Added &apos; stringsdict &apos; to FileTypes
Renamed EAP _ TTLS _ MD5 to EAP _ TTLS _ EAP _ MSCHAPv2 in RadiusProtocol .
Added test case for appending parameters with null values to HTTP request .
Fixed null pointer exception in PacketTransceiverImpl by checking if connection is null before calling write ( ) method .
Removed unused action code CUSTOM and reset ( ) method in AbstractHttp11Processor .
Added null check for InputMethodInfo in InputMethodManagerService
Added support for null type in ColumnValue
Renamed PlurkApi to MobilePlurkApi
Fixed PasteHandler to use DocumentUtil.executeInBulk for large pastes
Added -RT to search query and fixed double quotes issue in SearchedTrendsActivity .
Renamed AggregateLifeCycle to ContainerLifeCycle and updated import statements .
Added code to handle &apos; Channel ID ed419c76 ( outbound ) of Remoting connection 00052744 to / 127.0.0.1 : 4447 has been closed &apos; exception during remote stateful bean failover test .
Allow listener methods to be invoked with Method.invoke ( ) by setting accessible to true .
Added error logging for missing client queryable and timeouts in CachingClusteredClient and DirectDruidClient .
Added constructors for Drawer class
Added selected radio button to DetailListPreference
Remove unnecessary imports and ExecutorService usage in ConcurrentInstanceStartupIT
Replace &apos; getComponent &apos; with &apos; getContentComponent &apos; in CodeSourcePanel .
Added location search and view functionality to EditPostSettingsFragment .
Added check for container existence before checking block exists .
Added support for cross-validation in Job class by creating a new Job for CV and updating the job _ key .
Fix issue with offset column in SupervisedModelBuilder
Added support for writing media URI to parcel in MediaDescriptionCompat .
Removed unused variables and methods in ServerConfigImpl .
Added ProgressManager.checkCanceled ( ) in DataFlowRunner to ensure that the analysis is cancelled when requested .
Replace RemoteOperationProcessor with ResponseProcessor in BasicOperationService .
Modified StatementGenerator to use nameTable for type binding names
Update GPUImageLookupFilter.java to use highp instead of mediump for textureColor.b and quad1.y , and to remove unused variable &apos; intensity &apos; .
Added HTTP _ POST _ PROCESSOR to BeanIds and updated HttpSecurityBeanDefinitionParser to use it .
Renamed GLMLambdaSearchMetricsRace test class from hex to glm
Added support for inflating layouts with false flag to prevent auto-update of widgets .
Modified Checker.java to restrict module creation to only FileSetChecks .
Added exception handling for IllegalStateException in XActivityThread .
Remove unnecessary property setting in RuleDataMediumTest .
Added getLongExtra and getByteArrayExtra methods to ShadowIntent , and updated IntentTest to use them .
Added validateSchemaAgreement ( ) call in CreateKeyspaceStatement , CreateColumnFamilyStatement , and CreateIndexStatement to ensure schema agreement with the current version of the schema .
Added import for PartitionName and updated DocTableInfoTest to include PartitionName in column info
Added check for submodule with specific content in .git file
Added support for multiple classes in OrientGraphQuery by using a sub-query for multiple classes and fixed an issue with labels not being filtered correctly .
Added error logging for missing getter in private column access .
Remove unnecessary credit balance check in TestDefaultInvoiceGenerator .
Added try-catch block for closing output files and added verifiable and non-verifiable messages received counters .
Added check for null bucket segment and handling of spilled buckets in MutableHashTable .
Debug logging upgrade for HTTP2Connection.onUpgradeTo method .
Fixed bug in Table.setIdProperty method
Update PathChildrenCache to call updateInitialSet in all relevant cases .
Replace JavaUtilSet with JavaUtilHashSet in JUnitTestRunner .
Added @ ConditionalOnMissingBean ( RoutesCollector.class ) annotation to CamelAutoConfiguration.routesCollector method
Added a new method to PyUtil class to replace a statement list with a single pass statement .
Update dependency version for Forge from 7.8.0.684 to 7.7.2.682
Added download and add to classpath functionality for external URLs
Added test for non-embedded in-memory database shutdown
Added test case for PutRpc with infinite value and unsigned metric
Added getMaxParallelReplicaSyncCount method and set hazelcast.partition.max.parallel.replications property in Config .
Update AntScriptGenerator to use BuildConfig ( &quot; stbtruetype &quot; ) instead of BuildConfig ( &quot; test &quot; ) for StbTrueTypeBuild .
Removed unused imports and simplified code in ItemLaserTable.java
Added import statement for ApplicationManager in PluginManagerMain.java
Added initialization of _ state variable in PCGRNG constructor
Remove unnecessary removeCallbacks call in PowerManager.release ( ) method .
Added functionality to create a new account from the account selection activity
Added import statement for org.jetbrains.annotations.Nls and updated quick fix interface to use Nls.Capitalization.Sentence for sentence case .
Added support for case-insensitive JIDs by modifying the getJid method to use lowercase usernames and servers .
Removed print statement in CodeFormatterFacade .
Modified OSQLFilterItemField.java to use ODocumentHelper.getFieldValue ( ( ODocument ) iRecord.getRecord ( ) , name ) instead of ODocumentHelper.getFieldValue ( ( ODocument ) iRecord , name )
Temporary code for migrating 0.5 wallets to 0.6
Added imports for com.tinkerpop.blueprints.Direction , com.tinkerpop.blueprints.Graph , com.tinkerpop.blueprints.TestSuite , and com.tinkerpop.blueprints.TransactionalGraph , and added a new test case for OrientGraphSpecificTestSuite .
Added seek delta for fast forward and rewind buttons
Renamed JsFunction constructor parameter from \ &apos; getScope ( ) \ &apos; to \ &apos; getScope ( ) , &quot; jsCode &quot; \ &apos; .
Added support for serializing POJO classes that extend ODocument
Update InodeDirectoryTests to use Constants.SECOND _ MS instead of hardcoded values for last modification time
Added wildcard character to indicate prefix in ApduServiceInfo.java
Added nodeRefresh and edgeRefresh flags to TextManager &apos;s endRendering method .
Added variable &apos; currentKeys &apos; to PTFOperator.java
Added cdmaCheckAndProcessPlusCodeForSms method to handle CDMA plus code processing for outgoing SMS messages .
Added backup summary for WhatsApp
Removed unnecessary import of DimensionManager
Added check for empty question title before setting it to daily title
Added SLF4J bridge handler for Java util logging
Added support for RazorRepository .
Added ClockWithOffset implementation to allow for forced clock advancement during shutdown .
Added code to correct WH attachment in EnglishGrammaticalStructure and UniversalEnglishGrammaticalStructure .
Added support for converting Markdown links to HTML links in IpnbUtils .
Added debug logging for fatal I / O exceptions in NettyAsyncHttpProvider
Added import statement for android.os.DeadObjectException and handled DeadObjectException in TrustManagerService .
Added code to create storage directory path if it does not exist and set permissions .
Renamed &apos; speak &apos; method to &apos; say &apos; in TTSServiceFreeTTS class .
Added method &apos; defaultIfBlank &apos; to StringUtils class .
Added support for parsing ANTLR grammar with left recursion .
Remove unnecessary write of PARAM _ OPERATION in ModelControllerOperationHandlerImpl .
Added method kill ( ) to LocalTachyonMaster to interrupt the master thread .
Added parent requestDisallowInterceptTouchEvent ( true ) in swipe listener to prevent intercepting touch events .
Modified Serdes.java to change private class WrapperSerde to protected class WrapperSerde .
Fix for invalidating Graphite socket pool on error
Remove injector attribute when context is destroyed
Added equal ( ) method call to findFirst ( ) in tutorial class .
Added tinting for selected icons in PrimaryDrawerItem and SecondaryDrawerItem .
Added fields _ offset and _ weights to DataInfo class .
ZooKeeperStorePublisherTest : Changed ZKConnection constructor to use millisecond timeout
Changed logging level from ERROR to WARN for external diff and merge tools .
Added logging statement in SpringBatchProducer.prepareJobParameters method .
Update recognition service to be stateless
Minor change : Reduced time threshold for Search Everywhere action &apos;s &apos; ourLastTimePressed &apos; variable from 400 to 300 milliseconds .
Added method clearCustomData ( ) and updated removeCustomData ( ) to clear all key / value pairs from custom data field .
Added import statement for DumpDataType and set default dump data type to DUMP _ AS _ TEXT in LayoutTestsAutoTest .
Added import statement for Scaling class and updated Image constructor to use Scaling.none for checkbox image .
Modified DemoPlayer.java to add a new method onText and removed unnecessary abstract modifier from existing onText method .
Toggle SHOW _ STRUCTURE option in AnalysisUIOptions.java
Added logging for missing mapping in InputManager.deleteMapping ( ) method .
Added neutral button to Accounts activity for displaying full changelog .
Update fast scroll settings for MovieGridFragment
Removed unused import and simplified file color manager
Added static instance for DefaultClock and updated references to use it .
Added &apos; src &apos; parameter to &apos; addAll &apos; method in RelIdArray.java to improve performance .
Shorten menus by showing &apos; Animate Route &apos; only when controlling ongoing simulation .
Remove unused import org.apache.commons.lang.math.NumberUtils
Changed PojoUtils.FieldNaming from CONSISTENT to ORIGIN _ HAS _ UNDERSCORES and updated field name accordingly .
Defer logger initialization in SpringBootServletInitializer
Added assertTrue instead of assertEquals in AbstractTestQueries
Removed unnecessary System.out.println statement in TestELInJsp.java.
Added time unit to ThreadPoolProfileDefinition
Update copyright year to 2016 and add try-catch block for ClassNotFoundException in DefaultConfiguration constructor .
Added instrumentation package configuration option to AbstractInstrumentationMojo .
Removed unnecessary imports : ExecutorService and TimeUnit
Renamed ChooserTargetService to MyChooserTargetService and updated manifest file accordingly .
Added import statements for Spring Web framework and overridden buildRequestAttributes method to handle Grails web requests .
Renamed file path for backup index directory in DownloadActivity .
Replace newline characters with spaces in DebuggerExpressionComboBox
Added check for empty string in convertCase method .
Added logging to GroovyPagesServlet
Added Apache License header to PluginManager.java
Added check for non-nullable types in JSTypeRegistry
Added support for web authorization screen
Updated default value of DETAILS _ SHOW _ OPTION from false to true in CommitChangeListDialog .
Remove unnecessary constructor and setters for TaskConfig .
Added import statement for Serializable and implemented Serializable interface in PostLocation class .
Update AutoFactoryProcessorTest to use a different line number for the error message
Removed unused method AndroidDatabaseConnection.insert ( String , Object &#91; &#93; , FieldType &#91; &#93; ) and updated insert ( String , Object &#91; &#93; , FieldType &#91; &#93; , GeneratedKeyHolder ) to return the number of rows inserted .
Added support for multiple PSI files per document
Added check for fragment addition before launching reader subs
Added onContextItemSelected and onOptionsItemSelected methods to FilterListActivity .
Added nullability annotations to GenericDraweeHierarchy.java
Added trim ( ) method to decodeArray ( String encoded ) to remove leading and trailing whitespaces .
Added parsing of segments in isSelectionTemplate ( ) method .
Removed unnecessary parseInclusiveGateway method from CdiEventSupportBpmnParseListener .
Fixed assertions in RandomAccessDataFile.java to avoid unnecessary logs .
Added support for new browsers in Util.java
Added setTargetFragment method call to ProfileInputDialogFragment in MyProfileFragment .
Update ConstantCaptureActivity.java to calculate xpos using a floating-point value instead of a hardcoded value .
Added printing of time taken for sentences and sentences per second in Tools.java.
Added &apos; getAllPortlets ( ) &apos; method to &apos; LayoutTypePortlet &apos; class to return all portlets instead of only the ones that are currently visible .
Reset stream management session id to null when stream is cleanly closed .
Added handling of OOfflineNodeException and removed network connection if any .
Removed unused imports in Mapper.java and TypeHandlerTest.java
Updated InMemoryTaskArtifactCache.java to include a new logging eviction listener and updated the eviction mitigation message to reflect the appropriate process for reducing cache misses .
Reduced cleaner interval from 10000 to 1000 in DBApiLayer.java
Added test for getMaterial ( int id ) method with out-of-range input values .
Modified DBCollection.java to change protected abstract method insert to public abstract method insert .
Added configuration option for GRAB _ SIZE in MemoryManager
Added LinkedHashSet import and changed HashSet to LinkedHashSet in StandardRoot.java to retain the order of JARs returned .
Added foreign-fallback.html to ignore list due to flakiness
Added method &apos; query ( ) &apos; to interface Vertex for configurable querying of incoming and outgoing edges .
Fixed location update in SearchHistoryFragment
Added support for getting extension functions and ini settings through InfoFunctions class .
Added warning for emptying out wallet and added option to allow spending unconfirmed transactions
GoParserTestBase : Enable error checking in doTest method .
Added method getSource ( ) to TokenStreamRewriter class .
Added public modifier to DataInputInternal interface in DataIO.java
Added constructor with path parameter to ClientCredentialsTokenEndpointFilter
Removed unused variable &apos; index &apos; in FetchPhase.java.
Temporarily force crossModuleCodeMotion to run after property disambiguation for better optimization .
Remove unnecessary assertion in TypedTableExamples
Fixed issue with multiple delegating constructors in interface implementation
Update TableTest.java to remove unnecessary System.out.println statement .
Changed launchCheck _ checkbox text from &apos; Launch &apos; to &apos; Launch browser&apos;
Added onCompletion ( ) call in JCVideoPlayer &apos;s handleStateChange method .
Added combined-assets folder to excluded folders in AndroidFacetImporterBase
Modified AbstractNettyHandler to handle connection errors and propagate exceptions .
Replace FileReader with FileInputStream for reading log files
Update to use ASM4 ClassReader instead of ASM ClassReader
Added privateKeyPassphrase parameter to SFTP URL for deleting files
Added support for text-to-speech engines in TTSCommandPlayerImpl .
Update log level for handling events without a spring-event endpoint from WARN to INFO
Modified SimpleValueListener to use JsonXContent.jsonXContent.createParser ( ) to parse JSON values .
Updated logging level for GENA Subscription events from INFO to DEBUG
Added method &apos; sort &apos; to class Transaction and implemented it .
Added check for application info flags to determine if app is stopped or not before running backup .
Added import for VfsUtil and updated usage of output path to support Eclipse format .
Disable MORE _ DEBUG flag in BackupManagerService
Added field &apos; mContext &apos; to TreeWalker and updated visitors processing to use it .
Added logic to show pictures in SingleMessageView based on user preferences and contact list
Added deprecation warning for allocateRecidReuseEnable method in DBMaker class
Added ability to add rows to JBListTable and fixed isCellEditable method .
Update JythonBackend.java to use dsl.py instead of individual Python scripts
Added support for CACHED inMemoryFormat in MapService # equals .
Set require encryption instead of enable always in OtrContactMenu .
Flip buffer instead of rewinding it in OggStreamingSoundData.readNextInto ( ) method .
Added validation for unknown OWServer config keys
Modified AppearanceUIController.java to change creation of column observer to pass false as second argument
Set needsUpdate flag to false before adding Future callback to avoid race condition
Fixed JmxAutoConfigurationTests to correctly export MBeans
Updated JavaScriptTarget.java to add &apos; function &apos; to the list of reserved keywords and changed the return value of getVersion ( ) from &apos; 4.5 &apos; to &apos; 4.5 &apos; .
Added GNU General Public License header to BitmapCreationBenchmark.java.
Added border to ComponentWithBrowseButton for IntelliJ LaF and Darcula
Removed unnecessary MergeEvent class and updated constructor injection for BlockingCalculator .
Clear frame list and update timestamp calculation in FramesPanel .
Added method setQueryTimeout ( long millis ) to AndroidCompiledStatement , but it &apos;s not supported by Android API .
Modified ShinyAppsDeploy.java to change the selected index of the app list from Math.max ( 0 , apps.size ( ) - 1 ) to 0
Dispose debug session and variables in PyBaseDebuggerTask
Added logic to handle brackets and quotes in InputHandler.java
Added override of toString ( ) method in XMPPError class .
Added manual index configuration for OrientBaseGraph .
Update CameraDeviceBinderTest and CameraMetadataTest .
Replace CachingRegistry with DefaultRegistry in RegistryBuilder
ZLTextView : Changed getTextStyle ( ) .getLeftIndent ( metrics ( ) ) to storedStyle.getLeftIndent ( metrics ( ) ) and getTextStyle ( ) .getFirstLineIndent ( metrics ( ) ) to storedStyle.getFirstLineIndent ( metrics ( ) )
Added check for existing client before adding it to clients list in SocketIOConnection.connect ( ) method .
Added visibility check for overflow button in CardHeaderView
Added check for isFinishing before calling super.onBackPressed ( ) in AstridActivity .
Added import statement for com.google.inject.util.Modules and updated module loading to use Modules.override ( ) method .
Added logging for error connecting to server in PrimeConnections and HttpPrimeConnection
Added LICENSE file to platform file type factory
Added slash at the end of external storage directory path in getRootPaths ( ) method .
Add support for processing aliases in goog.modules
Added MIT License and copyright notice to MixinWorld.java.
Added try-catch block to handle RuntimeExceptions in DatabaseBackend class .
Added method &apos; dropTail &apos; to ContainerUtil class .
Added test case for capturing arguments with vararg methods
Added custom header view support to UltimateViewAdapter by adding a new method swapPositions that takes into account the custom header view .
Added support for US Gov West region in AwsEc2ServiceImpl .
Fix for unexpected errors during server startup by adding a quiet stop mechanism .
ExifInterface : added support for non-seekable file descriptors and input streams , and clarified API documentation .
Added build date and time to CasEnvironmentContextListener
Update ImmutableMethodImplementation.java to include registerCount in getRegisterCount ( ) method .
Disabled Synctex commands in PDFViewerPresenter
Update XMPPConnection.java to clarify documentation and remove unnecessary methods .
Fix bug in UnderFileSystem.java : Remove unnecessary null check for object
ui fragment DownloadButton visibility fix
Added new import statements for DAType , DataAccess , Directory , GHDirectory , and org.slf4j.Logger , and replaced baseUrl variable with a new value .
Added @ Lob annotation to field &apos; submitterNotes &apos; in class FieldTripRequest .
Added source paths step to JavaModuleType wizard
Modified ResolveInfo.java to change &apos; activity &apos; to &apos; activity or broadcast receiver &apos; in the comment and added a missing &apos; + &apos; before the &apos; * &apos; symbol .
Added deprecation annotation to MBeanUtils.getDomain ( ) method
Added non-persisted inodes to InodeTree # traverseToInodeInternal
Renamed internal database file and updated exception message for ClassID.java and StructDatabase.java
Set alwaysUseDefaultTargetUrl to false by default
Added check for existence of path before adding it to HDFS
Added &apos; true &apos; argument to DirectoryNavigateEvent fired when setting project directory .
Address.java : Changed ArrayList &lt; Address &gt; addresses = new ArrayList &lt; Address &gt; ( ) ; to List &lt; Address &gt; addresses = new ArrayList &lt; Address &gt; ( ) ; and updated null check for empty address list .
Fixed bug in RegexFilterFunction where env.buildTransitiveClosure ( ) was called with an incorrect argument .
Added logging statement for leanback launch intent in AppCompatibility test .
Added header filter strategy and message created strategy to SjmsComponent
Added test for concurrent create journal with fix for HDFS client caching issue .
Disable task log fetching in MR1 mode due to lack of TaskLogServlet support .
Added support for writing MIN _ KEY in BSONWriter .
Modified AppCompatDelegate.java to extend methods instead of using them directly .
Added Apache License header to CookieTest.java and removed unnecessary imports .
Change cancel ( ) to abort ( ) in SqlStageExecution
Removed unnecessary Builder constructor and sanitize method in VerbalExpression .
Update data sets and test cases paths to reflect new directory structure .
Update incremental search message handling
Added comment explaining the purpose and limitations of the segment tree class .
Added method enableAllTools and disableAllTools to InspectionProfileImpl
Fixed assertion in AbstractNativeCacheTest.java to use get ( ) instead of direct access .
Changed label from &apos; totalTime &apos; to &apos; requests &apos; in ScanProgressTableModel .
Added support for parsing multiple devices in ADB output .
Added &apos; Error &apos; to PluginResult enum and updated Status enum .
Added SparkJobMonitor logger to non-verbose mode LogDivertAppender filters .
Allow / remote.html access from any IP address
Removed unnecessary check for built-ins redeclarations in packages
Fixed issue 1346 : Set USE _ COMPASS _ IN _ NAVIGATION default value to true for CAR mode .
Added javax.servlet and javax.servlet.http exports to OSGIConfig
Remove unnecessary code - Use ALL available descriptions and not only Direct
Added call to &apos; myStatusBar.updateWidget ( ID ( ) ) &apos; in method &apos; update &apos; of class &apos; IncomingChangesIndicator &apos; .
Update SampleHypermediaJpaApplicationIntegrationTests to use absolute URL for location header
Added support for legacy &apos; new stories &apos; widgets in HackerNewsClient .
Added randomization to executor scheduled rate in HTreeMap and Store classes
Added handling of AssertionError in NioClientManager to prevent crashes on Android .
Added inheritance relationship between PsiClass and its superclasses in InferenceIncorporationPhase .
Added edge to ERROR state in DFA and updated addDFAState method to handle it .
Added line mEmptyView.setVisibility ( View.GONE ) ; and updated single page and auto expand preferences
Removed unused imports : PreviewImageActivity and PreviewImageFragment
Added Singleton class to hold reference to PrimitiveType values array
Added constructor for LiquidData that takes a LiquidStack and an ItemStack as arguments , and updated the existing constructors to use the new LiquidStack ( LiquidId , int ) constructor .
Removed unused import statement for CamelFactory in IntegrationTest.java
Replace FBUtilities with ByteBufferUtil in ColumnDecoder
Added case sensitivity option for table cache in MySQLCatalog
Typeface.java : Fixing create ( ) method to handle invalid style values .
Fixed OAuth2Authorization.java to correctly handle null tokens .
Update TAG from &apos; AudioRecord-Java &apos; to &apos; android.media.AudioRecord &apos; and remove unnecessary Log.e statements .
CAMEL-2824 : Avoid deadlock when aggregating and timeouting exchanges
Dispose of video renderer when replacing video track
Added public modifier to &apos; analyst &apos; field in CommandLineParameters.java
Added spatial index on latitude and longitude columns in OpenCellID database for efficient querying .
Fixed bug in ImmutableInstruction20t.java by changing ( byte ) instruction.getCodeOffset ( ) to instruction.getCodeOffset ( )
Added support for let and const declarations in externs
Added support for on-the-fly inspections in JavaFxUnresolvedFxIdReferenceInspection .
Modified ModTextureAnimation and ModTextureStatic classes to use tileSizeSquare instead of TextureFX.iconTileSize
Fix bug in HttpResponseDecoderTest : Garbage input should generate the 999 Unknown response .
Added @ Contract annotation to deparenthesize method in JetPsiUtil.java
Update InfinispanSubsystemXMLReader _ 1 _ 1 to use StartMode enum and fix listener executor attribute
Added license header and moved ViewConfiguration code to ShadowViewConfiguration.java
Added methods acquireWakeLockWithUid and crash to BridgePowerManager interface .
Renamed &apos; createAlias &apos; method to &apos; getSubjectField &apos; and updated implementation to use new CertificateWrapper class .
Changed lock from shared to exclusive in ODefaultCache # get and # keys methods .
Modified GrTypeDefinitionImpl to use add ( ) instead of addBefore ( ) for adding elements to the body .
Renamed &apos; requestAction &apos; to &apos; action &apos; in PluginOperationResult and updated switch statement in PluginCommand to use the new field name .
Fixing issue with TitlePageIndicator &apos;s current page being out of bounds .
Update copyright year in AnnotationJmxAttributeSource.java from 2007 to 2009 .
Added Apache License header to RetryingScheduledThreadPoolExecutor.java.
Added import statement for &apos; org.jf.smalidea.SmaliIcons &apos; and added method &apos; getElementIcon &apos; to override the icon for Smali classes .
Added event handlers for goToFileFunction and reflowComment commands in ConsolePane .
Replace random string with repeated &apos; A &apos; string in HttpConnectionTest .
Fixed issue with F1 score calculation in Evaluation class .
Added method getNullables ( ) and getNotNulls ( ) to class NullableNotNullManager .
Added method exists ( int userId ) to UserManagerService to check if a user exists .
Added remove method to BroadcasterFactory and DefaultBroadcasterFactory
Added test for Model Parcelable
Throw more specific exception types in WalletProtobufSerializer
Added import statement for com.intellij.openapi.application.ApplicationManager and updated saveDocumentsIfNecessary method to check for Application disposal .
Removed unnecessary buffer expansion in SecureNioChannel constructor .
Added method build ( Project project , Language language ) to return a LanguageConsoleView for the given language .
Fixed case sensitivity issue in ShadowSQLiteCursor.java
, exponentially biased towards newer entries .
Added try-catch block to close ( ) method in DbSqlSession.java to handle exceptions and prevent them from being swallowed .
Added methods isRoleSearchAsUser ( ) , setRoleSearchAsUser ( ) and searchSubtreeForRoles ( ) to JNDIRealm class .
Updated JglfwApplicationConfiguration class with new default values for foregroundFPS and backgroundFPS .
Changed SCROLL _ MULTIPLIER from static final float to private float
MagicNumberCheck : Allow final and interface constant definitions .
Remove unnecessary &apos; apply &apos; method in CharMatcher
Changed threshold for detecting unstable weights and biases in DeepLearningModelInfo .
Reduced neighbor update percentage from 40 to 10 in PrepareContractionHierarchies .
Added import statement for IStubElementType and added constructor for PyClassImpl that takes IStubElementType as parameter .
Fixing error message for GrailsTagException in ParseTests.java
Debug logging for GCalEventDownloader when given event content doesn &apos;t match regular expression
Added method getChangeLogParameters ( ) and getDatabase ( ) to class Liquibase .
Added functionality to save password for BaseAuthDialog .
Fix for PyPropertyDefinitionInspection : removed unnecessary check for call site and improved type evaluation
Added logging for discarded resolve failures in UserResolverChain .
Added setRefreshDrawable method to PullRefreshLayout .
Changed color scheme for patch apply status
Fix for DexterInstanceTest : updated checkPendingPermissions ( ) to checkPendingPermission ( PermissionListener permissionListener )
Modified DBCollection.java to remove unnecessary &apos; options &apos; parameter from find and rename methods , and updated documentation to reflect this change .
Added logic to handle non-Hystrix context in HystrixNetworkAuditorEventListener .
Added support for overriding elevation profiles for street edges and vertices .
Renamed &apos; Loading recent history &apos; to &apos; Loading History &apos; in VcsLogDataHolder and VcsLogRefresherImpl .
InplaceRefactoring : Changed injectedToHost method to shiftRight ( shiftOffset ) before checking containment .
Remove unnecessary print statement in XmlReader.java
Added AnalyticsTracker import and updated NetworkUtils import in ReaderPostListActivity.java and ReaderTaskFragment.java
Modified query in getPeopleCountForLocalBlogId method to use AND instead of &amp; for concatenating WHERE clauses .
Added UUID random generation for offline player testing
Update ReaderPostListFragment to use transparent action bar and remove unnecessary imports .
Fixed bug in Polygon.java where localVertices were not being updated correctly when dirty flag was set .
Added @ Autowired annotation to ResourceServerTokenServicesConfigurationTests .
Added volatile qualifier to _ readOnlyMap field in SerializerCache class to ensure correctness in concurrent access scenarios .
Replace LOG with _ log and remove unused import
Removed unnecessary import statement and logger statement in SearchController .
Fixed trim ( ) method call in getRelativeFileHandle ( ) method .
Fixed scaling issue in IconUtil by translating the graphics context and using bicubic interpolation .
Added support for recursively serializing fields from superclasses in AnnotatedCompositeSerializer .
Update parameter name from &apos; JsonSchema &apos; to &apos; Schema &apos; in method &apos; writer &apos; of class &apos; ObjectMapper &apos; .
Update CollectionAcceptanceTest to use find ( ) instead of one ( )
Update LocalFolderPickerActivity to extend ToolbarActivity and add toolbar setup .
Modified MediaDriverFacade interface to change library to client and updated comments
Fixed bug in ComponentInterceptorAnnotationParsingProcessor : now correctly processes ExcludeClassInterceptors annotation
Added &apos; connectionType &apos; and &apos; requestType &apos; to WXPerformance.getDimensions ( ) and updated WXPerformance.getMeasures ( )
Update WebSocketServerHandshakerFactory to use a different URL and sub-protocol request .
Fixed issue with null FilePathImpl in TreeModelBuilder .
Added support for BlockingIOCometSupport in DefaultAsyncSupportResolverTest .
Renamed FakeTaskScheduler from sample / src / androidTest / java / com / karumi / rosie / sample / testutils to rosie / src / test / java / com / karumi / rosie / testutils
Added omitted method checking for StandardContext
Update ReaderPostListFragment to show posts with a specific tag or in a specific blog
Added check for framework APK existence before falling back to default APK
Added full message field to SyslogEventHandler
Added import statements for StringTools and updated ManualTagger to use it .
Update date format in WebkitTest and BackupManagerService
Updated HgMqUnAppliedPatchesPanel to use HgQRenameCommand.performPatchRename in the background
Added test for filtering non-XML chars in JAXB marshaling
Fix issue with ObjectMapper setting incorrect value for fields .
Added activation of database on current thread in OSQLNonBlockingQuery.run ( ) method .
Renamed &apos; googBase &apos; to &apos; useOfGoogBase &apos; in DiagnosticGroups .
Modified LogAspect to use org.jasig.cas instead of org.jasig1.cas
Added Future.get ( ) call to ensure index recovery completes before moving on to next test case .
Added check for null &apos; node &apos; attribute in DiscoverInfo and set it to request node if necessary . Also , fixed bug in calculating entity caps string .
Added debugging functionality to DefaultCodeFactory by tracking the position of the current scope and breaking out of the loop when no more codes can be generated .
Added constructor to ComponentDao for dependency injection
Added wheel and delta wheel information to MouseMotionEvent.toString ( ) method .
Updated RequestSpec to return &apos; this &apos; instead of void for method ( ) and set port default value to 5050 in LaunchConfigBuilder .
Added logging statements and improved type-safe casting for BSONSerde
Added call to &apos; modelView.close ( ) &apos; in ModelNodeInternal.java
ZoomManager : updated post-scale logic to account for rotation and fixed viewport
Added check to prevent reloading props files unnecessarily in DefaultAppCore .
Util.java : Made attachRecords method final and removed unnecessary null check for settings object
Removed unused constructor for WebappClassLoader
Update SettingsPopulation.java to change read-lookback option &apos;s description and add no-wrap option .
Fixed priority comparison in SimplePluginManager &apos;s event listener registration .
Remove unnecessary call to clearAppProfilesLIF ( ps.pkg , UserHandle.USER _ ALL ) ;
Replace QPTreeTransformer.QPtransform with QPTreeTransformer instance .
Allow multiple expires directives with optional values in ExpiresFilter
Fix typo in method name : &apos; testEqualsWithItSelf &apos; - &gt; &apos; testEqualsWithItself&apos;
Remove unused variable &apos; useNameReferenceGraph &apos; in PureFunctionIdentifier.Driver constructor .
Fix for bug # 12345 : Handle non-text messages in WsRemoteEndpoint .
Removed unused import statement for &apos; javax.annotation.PostConstruct &apos; in OpenIdPostUrlHandlerMapping.java
Added singleton instance of Some &lt; Void &gt; and added method nothing ( ) to return it .
VcsInitialization : Use ApplicationManager.getApplication ( ) .executeOnPooledThread instead of StartupManager.getInstance ( myProject ) .registerPostStartupActivity
Added additional startSillyBench ( ) call and changed getCounter ( ) return type from long to int .
Added method findAll ( Class &lt; T &gt; modelClass ) to DataSupport class .
Added logs for cases where BeaconService is not connected while adding or removing regions .
Added imports for IndexRoutingTable , IndexShardRoutingTable , and ShardRouting
Added static import for anyBoolean ( ) in PluginDownloaderTest .
Update Spanish Tagger test to use new tagging scheme
Added method setTextSize ( int , float ) to class FontAwesomeText and updated setTextColor ( int ) method .
Reset RobolectricGen to use RobolectricBase instead .
ZAP : Added note to HttpMessage and support for multiple tags , also added historyRef and logger
Fixed bug in TvInputService and TvView
Update getPreferredSize ( ) method in ActionToolbarImpl to handle empty bounds list
Added constructor with title parameter to DefaultValueColumn
Renamed SetHeaderHandler.Builder to Builder for consistency with other handler builders .
Added FAB _ SIZE annotation and updated imports
Added test case for iterator remove method .
Added GetConnectedClientsOperation for retrieving connected clients .
JetPositionManager : Changed createCachedValue computation to use ResolvePackage.getAnalysisResultsForElements instead of ResolvePackage.getAnalysisResults
Added synchronized method reset ( ) to ConnectionState class .
AsyncFeatureTransferManager : updated retry logic to wait for a maximum of maxWait milliseconds and a maximum of maxRetries attempts .
Handle InterruptedException and ExecutionException in Cache.java
Modified DirectoryUtil.java to change the error message for BerkleyDB storage to local storage .
QSPanel : Only hide content when still in detail state .
Fix rotation changes and update time column width calculation .
Replace AbstractGreeter with GreeterImplBase in ErrorHandlingClient
Modified log message for payment method retrieval error
Removed unnecessary call to soloActivity.waitForIdle ( ) in Clicker.java
Renamed method createFileWriteAction ( File output , String encoding ) to createFileWriteAction ( File output , String encoding )
Added support for time counters in QueryFragmentCounters
Removed unused components from FirstPersonClientSystem
Added a brief delay in TestWsWebSocketContainer to ensure that the WebSocket session is closed properly .
Fixed cloning process for JmeCloneable objects by calling cloneFields ( ) method after creating a shallow clone .
Ensure can create file by checking canWrite ( ) instead of exists ( )
Added constructor for UsernamePasswordAuthenticationCredentials
Update MockRestAdapter.java to wrap mock implementations of API interfaces with delay and error characteristics .
Added RequestContextFilter and enabled favicon configuration .
Change first month day from 1 to 15
Fixed issue with sourceSets.getMainSources ( ) .create ( name , type ) returning null instead of the created object .
Added checks for targetWidth and targetHeight to ensure they are non-negative before taking a picture .
Added onCompletionListener null check in OpenALMusic.java
Added support for ZipFile in IOResourceInspection
Fixed issue with document indent options not being reinitialized properly in viewer mode .
Fixed issue with null comments and IMDB IDs in PrepareRequestTokenActivity .
Added settings for spaces around key-value delimiter and blank lines in properties files .
Scrubber : updated key comparison to handle null keys and avoid unnecessary exceptions .
Updated JNDI lookup for TransactionManager and TransactionSynchronizationRegistry to use &apos; java : jboss &apos; instead of &apos; java : / jboss &apos; .
Added method toString ( ) to CoreProgressManager and ProgressManagerImpl
Fixed issue with scrolling while refreshing
Modified ModuleEditor to use &apos; module &apos; instead of &apos; myModule &apos; for getting components and extensions
Added test for global entities stored and restored
Added copy of token labels to extentTokens in RuleBasedCorefMentionFinder to avoid parser issues .
Added support for circular ripple selector and activated state to color chooser .
Added _ _ module _ _ attribute to PyStructureViewTest .
Fixed race condition in javaURLContextFactory by synchronizing access to initialContext .
Added support for &apos; Operation.CREATE &apos; in DocWriteResponseTests
Update GroupProperties reference to GroupProperty in Operation class .
Updated Form.java to use setAnswer ( String , String ) for non-basic types .
Fixed issue with PageIndicator by adding a check for null before laying views on top of each other .
Fix TypeUtils resolveTypes method to use TypeManager instead of creating a new instance every time .
Update expect class from IOException to Exception in test method getResponse
Added closureRewriteClass check to DefaultPassConfig
Upgrade JTASessionContext to use ConcurrentHashMap for currentSessionMap
Added SonarSource copyright and license information to package-info.java.
Added public version of notification content view
Modified LdapUserService to create user objects using simple username and updated createUserFromLdap method to accept additional parameter .
Rename test method to indicate it &apos;s a long-running test
Set Filename method now checks for null or empty string before setting the filename .
Updated CamelNamespaceHandler to use default auto-generated name for contextId when no explicit id is set .
Added support for non-power-of-2 framebuffer textures in LwjglRenderer .
Added parameter type hinting for Queryable interface
Disable standard extensions to ensure javac is loaded from the right tools.jar
Fixes # 12345 : Changed TimestampCheckingOffset calculation to use &apos; input &apos; instead of &apos; timeStart &apos; for better performance .
Added code to disable PrivateFactory injection in CalligraphyConfig .
Added support for inner polygon types in BinaryMapIndexWriter .
Added check for title and items before setting content in MaterialDialog build method
Updated DocumentsContract.java to add FLAG _ DIR _ SUPPORTS _ CREATE flag .
Removed deprecated synchronized function and updated imports .
Renamed handleReplicaSetMemberChange to handleReplicaSetMemberChanged
Added deferred start of SyncJobService to avoid ANR
Added constructor for OpmlImportWorker with Context and Reader parameters .
Removed PATCH verb from Verb enum
Added support for generic multi-value in ODocumentHelper
Added import for org.jetbrains.jet.lang.descriptors.annotations.Annotated
Added import statement for org.jetbrains.annotations.NotNull ; and changed getModuleModifiableModel method to use @ NotNull annotation for parameter Module
Added registerPlugins ( ) and onAfterActivate ( ) for lifecycle listeners in OServer constructor .
Added documentation to RuleFilenameException constructor explaining the naming convention for rule files .
Replace path separator characters in display name to avoid inconsistent paths .
Deprecate REUSE _ RULES _ CONFIGURATION _ PROPERTY and getReuseExistingRulesConfig ( ) method .
Modified MainActivity.java to adjust FloatingActionButton margins for devices with navigation bars .
Fixed an issue with memory reservation where newReservation would sometimes exceed maxMemoryReservation
Added import statement for android.text.TextUtils and updated usage of StringUtils.join to TextUtils.join in setAutodownloadSelectedNetworks and setHiddenDrawerItems methods .
Added try-with-resources block to properly close server controls in InProcessServerControls.java and Neo4jRule.java
Added support for OPTIONS and HEAD requests in RestController .
Added support for Oracle and PostgreSQL databases in MigratorTest .
Added error handling to MasqueradingPsiBuilderAdapter
Modified GetListActiveUsers.java to change the type of the &apos; activeuserjids &apos; variable from &apos; jid _ single &apos; to &apos; jid _ multi &apos; .
Added preferred focused component and log categories support to DebugLogConfigureAction .
Added support for determining whether the device supports encryption , even if it &apos;s set to default .
Fix issue with incorrect page size calculation in StoreCached.java
Update AbstractDatabase.java to use escapeTableName and escapeStringForDatabase for SQL injection protection .
Added check for action buttons in ActionMenuPresenter
Update to use StdFileTypes.XML instead of hardcoded file extension in createTreeFromText method .
Modified ODefaultReplicationConflictResolver to correctly log conflict records with current and other version numbers .
Fix issue with null downloader argument in Picasso builder
Change performDexOptIfNeeded to performDexOpt with false value
Added support for extracting script subtags from locales , and updated tests to include cases for locales without a real language .
Removed unnecessary null check for KernelStatement constructor
Fix error handling in JetTypeMapper
Update copyright year in DnDManager.java and simplify getInstance ( ) method .
Added nullability annotations to JetContentBasedFileSubstitutor .
Added support for lambdas in predefined configuration .
Added firewall rule for power save temp whitelist apps
Removed unused variable &apos; ActiveTasks &apos; from TpStatsPrinter.java
Added userLogin field to IssueComment and updated IssueUpdater to include new severity update logic .
Added support for ZMQ _ REQ _ CORRELATE and ZMQ _ REQ _ RELAXED options in Options class .
PluginModuleType : Added isValidSdk method and updated imports .
Added test for placeholder in placeholder value and updated message replacement tests .
Allow remote process creation with true instead of false .
Minor change in SamlCompliantLogoutMessageBuilder.java : LOGGER.debug ( &quot; Generated back-channel logout message : &#91; { } &#93; &quot; , logoutRequest ) ; changed to LOGGER.debug ( &quot; Generated logout message : &#91; { } &#93; &quot; , logoutRequest ) ;
Fixed argument mapping for PyArgumentListImpl
Added logging for retrieving subtypes of a PSI class
Refactoring : Allow copying of non-synthetic classes only
Update SimpleHandlerResultHandlerTests to use Reactor Flux instead of Reactor Fluxion .
Fixed issue with result set viewer panels visibility
Added method getScrollListener ( ) and changed variable mScrollListener to private .
JSONEnumConvertor : Use Enum.name ( ) instead of obj.toString ( )
Remove unnecessary import statement for &apos; Settings &apos; in DocSchemaInfo .
Added logging statement for non-empty own device list in AxolotlService.registerDevices
Added TemporaryJobs builder for HeliosSoloDeploymentTest
Added check for activity being null before updating favorites in DashFavoritesFragment .
Added import statements for DocStringFormat and PyDocumentationSettings , and updated testDocstring to use REST format for docstrings .
Added static final variable MAX _ INSTRUCTIONS _ SIZE _ TO _ OPTIMIZE and added a check for it in OptimizationMethodVisitor.java
Replace &apos; ConversionException &apos; with &apos; CheckstyleException &apos; in FileLengthCheckTest.java.
Added performance test for direct use of Sequencer without RingBuffer
Added clearing of possibly existing headers in copyHeaders method .
Fixed issue with getting destination folder name in ChooseFolder activity .
Changed auto save interval from 20 seconds to 5 seconds in SystemConfig.java
Added exitWithSuccess ( ) call in Main.java
FList : Fix comparison of head element with null
Added truncate method to SinglePartitionSliceCommandTest
Fixed issue with Watcher.java where filePathName was not correctly set for sync accounts with missing folder .
Remove unused &apos; avoidsConfiguringTmpDir &apos; method and property from GradleDistribution .
Increased RENDERER _ COUNT from 5 to 6 .
Added support for icon customization in quick fix popup menu
Modified MessageHistoryServiceImpl to use opSetSMS instead of opSetIm for message listening .
Added support for retrieving a list of active loggers in the logging configuration .
Fix for domain class file annotation
Update ImageMessageFlyweight.java to use &apos; offset + CHANNEL _ OFFSET &apos; instead of &apos; CHANNEL _ OFFSET &apos; when getting the length of the channel name .
Fixed bug in IteratorIterable.java where iterator was not being properly reset
Fixed typo in OpCommand.java : &apos; Oping &apos; - &gt; &apos; Opping&apos;
ContactsContract : Changed parameter &apos; lookupUri &apos; from &apos; Contacts # CONTENT _ LOOKUP _ URI &apos; to &apos; ContactsContract.Contacts # CONTENT _ LOOKUP _ URI &apos; for consistency with other method signatures .
Update DBCollection.java to use wrap ( ) instead of wrapAllowNull ( ) for criteria and projection
Added function to add governor arc labels to basic graph .
Added import statement for WPWebViewClient and updated ViewPostFragment to use it .
Added OS-specific test for StrictMath.hypot ( ) method .
Added SELinux enforcement status field to Settings class .
Remove unnecessary download of descriptor when latest.lowestStatus is requested
Removed unnecessary imports and simplified createStreamFromRequest method .
Fix for missing &apos; version &apos; element in FMLModContainer.java
Added documentation for RecyclerViewDragDropManager &apos;s attach and detach methods .
Added video link and updated text justification algorithm .
Added import statements for android.util.Log and updated CompassListener class to include onReset method .
Added Guice-friendly Provider for JSR-330 Providers
Added synchronization to XResourceLoader.initialize ( ) method .
Remove unused method and fix cache size calculation .
Added System.getProperty ( ) calls to retrieve initial token , storage port , and rpc port values from environment variables , instead of hardcoding them in the configuration file .
Replace async task execution with AndroidUtils.executeAsyncTask
Added support for requesting available capabilities in CameraCharacteristics .
Added Google Analytics event tracking for show count .
Update copyright year in ConsoleProcess.java from 2012 to 2017 .
Extract inner element for fixed-ratio widgets in ChunkOutputStream
Removed unused methods setLayoutDirection and setLevel from LayerDrawable .
Added code to write the resulting manifest to the output directory , maintaining full path to prevent collisions .
Added support for determining SIM provision status based on CSIM app presence .
Added break statement in NioWorker cleanUpWriteBuffer method to improve readability and maintainability .
Fixed issue with incorrect type returned by toStringMethod invocation in DebuggerUtils.java
Added support for Grails tags in GSP pager
Added file / cache / recovery / last _ kmsg to Dropbox for system recovery log
Fixed issue with named broadcasting
Update version number to 8.5.0 in Constants.java
Remove unnecessary dependency on jboss-as-controller artifact .
Changed default value of DELAY _ TO _ START _ NAVIGATION from 10 to -1 and added a new getDefaultValue ( ) method to return 10 for CAR mode and -1 otherwise .
Added &apos; TODO SF needs a rename job &apos; comment in IncrementalJavaCompiler.java
BinarySplitLayoutPanel : updated logic for calculating splitter position
Remove dead node check in ZWaveConverterHandler
Disable testTransferMigratedSubscriptionWithCTDInFuture
Removed VersionedValue.STATUS _ LEFT from SILENT _ SHUTDOWN _ STATES in Gossiper.java
Update HystrixThreadPoolProperties.java to use &apos; withCoreSize &apos; and &apos; withQueueSizeRejectionThreshold &apos; instead of &apos; setCoreSize &apos; and &apos; setQueueSizeRejectionThreshold &apos; .
Added validation for sub-characteristics compliance
Fixed exception handling in ONetworkProtocolBinary and ONetworkProtocolDistributed
Added support for FirebirdDatabase in DoubleType
Added Apache License header to Whitelist.java.
Set chat pager save enabled to false
Added forceUpdateVisibilities method to NotificationContentView and called it when mForceSelectNextLayout is true
Added method createEntity ( D arg ) to AliasAwareExprFactory , which returns a PEntity &lt; D &gt; object and also added @ SuppressWarnings ( &quot; unchecked &quot; ) annotation to avoid warning for unchecked cast .
Added support for multiple tracks in MediaMuxer , allowing for simultaneous encoding and muxing of audio and video streams .
Removed unused import statement for android.util.Log in NativeBitmapFactory.java and MainActivity.java.
Fixed CollectionBinding.java to use TreeSet for sorted sets
Removed unused methods getThroughput ( ) , getHighestThroughput ( ) , and resetThroughput ( ) from MessageCounter interface .
Added validation end listener and styled button for invite functionality in PeopleInviteFragment .
Fixed EditorComposite &apos;s requestFocus and requestDefaultFocus methods to properly handle null component cases .
Removed unnecessary logs and simplified multicast receiver thread code .
Renamed variable &apos; algorithm &apos; to &apos; algorithmName &apos; and updated constructor to use &apos; getMessageDigest ( algorithmName ) &apos; instead of &apos; getMessageDigest ( algorithm ) &apos;
Added support for font smoothing in PGraphicsJava2D .
Added punctuation to output in AnnotationOutputterITest .
Fixed detection of encoding for book IDs
Added check for operator stats being done before closing exchange client .
Added check for SocketState.CLOSED in AbstractProtocol.java
Added support for parsing multiple signing identities from security output .
Fixed texture rendering for pipes by using ForgeDirection.UNKNOWN instead of hardcoded directions .
Added SkylarkCallable annotation to Platform.getNameInPlist ( ) method .
Update JUnit38ClassRunnerTest to use assertEquals instead of Assert.assertEquals and remove unnecessary annotations .
Removed unnecessary imports in SamlIdPProperties.java.
Added method isButtonEnabledForHistoryReference and isSafe to class PopupMenuCopyUrls for better functionality .
Modified RemoteConnectionService to handle disconnected state and conferenceable connections .
Fixed bug in HBaseKeyColumnValueStore.java : storeManager.mutateMany ( ) now uses storeName instead of columnFamily .
Added support for Java 8 &apos;s ZonedDateTime in PrimitiveType enum .
Modified method buildFormDescriptor in class DefaultPaymentGatewayApi to use custom fields instead of merged properties .
Added method getAssociatedObject to BaseAdapterHelper
Added fields for account name and type to Calendar entity .
Update GuardedByChecker to use MATURE maturity level instead of EXPERIMENTAL .
Added logging statement for dead process state
Added generic type parameter to ClassReflection.newInstance ( ) method .
Increased maximum number of folders in VcsStructureChooser from 10 to 100 .
Added &apos; try &apos; to the list of keywords in CheckUtil.java.
Added import statement for org.apache.commons.lang.StringUtils and updated match.getMessage ( ) to use StringUtils.abbreviate ( ) to truncate long messages .
Removed unnecessary import statement for &apos; SIZE _ OF _ INT &apos; in AeronTest.java.
Fixed incorrect matrix multiplication in Matrix3f.java
Modified Tomcat.java to add / remove URL package prefixes .
Fixed error in system.listMethods response validation
Replace slashes and linebreaks in JavadocOptionFileWriterContext .
Added deprecation for ExponentialBackoff ( int , int ) constructor and updated the documentation .
Added import for GroupNames in BooleanMethodIsAlwaysInvertedInspection.java
Update copyright year and add new author Florin Patan .
Release editor in project closing .
Added constructor for DeleteIndexedScriptRequest
JdkUtil : updated classpath handling to use UrlClassLoader and added support for command-line wrappers
Removed unnecessary ApplicationManager.getApplication ( ) .runReadAction call in CompletionService .
Modified XmlLog.java to use a simpler approach for formatting XML logs .
Removed unnecessary license header from SoundTest.java
Added import for InjectedLanguageManager ; fixed scope update for injected languages
Added ClusterState import and used it to get nodes IDs for list method
Element.java : Changed attribute retrieval for &apos; id &apos; from &apos; attr (
Modified GenerateServiceTicketActionTests to use AbstractCasWebflowConfigurer.EVENT _ AUTHENTICATION _ FAILURE instead of &apos; error &apos; for failed authentication attempts .
Modified Settings.java to update the description of the ANDROID _ ID constant .
Added start delay to TransitionSet for sequential transitions
Added exact matches for participle tagging in MultiWordPreprocessor .
Added getStatusMessage and getStatusMode methods to Editor class , and made EditorStatus constants public .
CollisionResult compareTo implementation changed from manual comparison to using Float.compare
Update ReportsFragment to use empty string for root region download name
Update copyright year in ListTemplatesHandler.java and add import statement for ApplicationManager
Added import statement for Desktop class in DomUtils.java.
Removed unnecessary imports : PollingConsumer and UriParam
Added check for unsupported MIME types in HttpConnection.java
Removed unnecessary setIconAttribute call in ShutdownThread .
Moved command text field to north in GoCommandConfigureDialog
Disable auto-zoom when map is dragged
Replace Pattern.compile ( ) with String.contains ( ) in Matchers.java
Added execution of enqueued actions on every traversal in case a detached view enqueued an action
Fixed ZenCodingUtil.checkFilterSuffix ( filterSuffix ) to return correct value .
WebSocketSession.java : Changed LOG.ignore to LOG.warn for exceptions on end-user WS-Endpoint .
TileMachine.java : Changed powerRequest ( ) method to return activation energy instead of max energy received .
Fix issue with double slashes in FtpUtils.java
Added FontAwesome CHECK and EXCLAMATION _ TRIANGLE icons
Removed unused import &apos; com.mongodb.WriteConcernResult &apos; in MongoCollection.java
Fix bug in FakeFile constructor
Added TokenTypes references to Check.java
Added OCFile targetFile to FileDetailActivity &apos;s intent
Remove unnecessary @ Autowired annotation and move initialization logic to a separate method
Removed handling of duplicate or out-of-order rows in RapidsTest .
Added support for reading client version from atmosphere.properties file .
Removed unused import com.mchange.v1.util.ClosableResource
Throw IllegalStateException if type is not LITERAL or INTERPOLATION
Added support for SVG images in HTMLFormatter .
Update KotlinRuntimeLibraryCoreUtil to use kotlin.Unit instead of kotlin.jvm.internal.Intrinsics
Added import statement for org.neo4j.com.ComException and updated TransactionCommittingResponseUnpacker to catch and handle IllegalStateException .
Added import statements for Handler and Looper , and updated search text timer trigger to use Handler to post a runnable on the main thread .
Renamed MocoJsonSocketHttpRunner.java to MocoJsonSocketRunnerTest.java and updated import statements .
Added try-catch block for handling illegal argument exception when retrieving cluster by name .
Added idempotence guarantee for ExecutionList.run ( ) method .
Replace NewFalsePositiveNotificationDispatcher with DoNotFixNotificationDispatcher
Added imports for android.graphics.Rect and android.graphics.drawable.Drawable , and updated seek bar and progress bar backgrounds to use custom drawables .
Fix worker metrics name validation
Fixed incorrect pause location reporting in DebuggerTestCase .
Added support for UTF-8 encoding in AESCrypto.java
Update prune cutoff days from 45 to 15 in SecretContentDAO .
Updated RequestServer.java to improve caching by using _ cache.get ( uri ) instead of null
Added implementation of CometProcessor interface
Added methods for setting optimization options : setUseTypesForOptimization and setRewriteFunctionExpressions .
Added @ Ignore annotation to TestXa.testDynamicRecordsInLog ( ) to suppress failure
Update AnnotationParameterInfoHandler to handle comma and closing parenthesis correctly .
Added deprecation warning to EntryInfo constructor and added new constructor with CharSequence parameter .
Added methods for getting class location and Jodd location .
Remove request for Window.FEATURE _ ACTION _ BAR in ActivityController
Update to use support.v7.view.ActionMode instead of android.view.ActionMode
Fixed Material.java to correctly guess render state apply based on defaults
Added removal of long press callback in ACTION _ CANCEL case .
Added method checkIfApplicationIsSuspended to RestrictedLockUtils , which checks if an application is suspended and returns the enforced admin component and user details if it is .
Renamed method &apos; setup &apos; to &apos; setUp &apos; in AbstractTestTag.java
Removed unnecessary resource from deployment annotation .
Reorder result set : remove forced reordering when removing ordering on some column
Added support for generating float and character literals in StubGenerator .
Added documentation for getSeaLevel ( ) method .
Fixed issue with empty street name and reference in MenuTitleController
Remove @ SuppressWarnings ( &quot; deprecation &quot; ) annotation from getScrollListWindowHeight method in ViewFetcher class .
Added logging statement to TsdbQuery.java to handle unexpected dropping of spans
Added a new line reading mechanism in FullyExecutableJarTests .
Added support for processing Thermostat Mode Get reports in ZWaveThermostatModeCommandClass .
Modified ParcelableGenerator to use public static final CREATOR instead of private static final CREATOR .
Added test for date literal expression
Modified IOSMethodBinding.java to change the implementation of the &apos; overrides &apos; method to also consider delegate equality .
Added ProgressListener to NettyAsyncHttpProvider for tracking write progress .
Added profiling options for Groovy compiler
Improved default values for boolean properties in Defn interface .
Set Lock Task Mode for non-system initiated tasks
Updated StructStrategy to ignore synthetic methods .
Added test for PlistSerializer.serializeToString ( ) and updated assertions in existing test .
Added import of com.google.common.jimfs.Configuration and changed FileSystem constructor argument to use Configuration.unix ( ) for testing java.nio.file Path behavior .
Added documentation for CameraProperties.SCALER _ AVAILABLE _ PROCESSED _ SIZES
RESTDocsGenerator : updated getFW method to use AsciiDocGenerator &apos;s getFW instead of hardcoded file path .
Added OWNERSHIP privilege to HivePrivilege enum .
Moved method formatTimestamp to be private and added a new private method with the same name to SerDeUtils .
Added ESCAPE key binding for cancel action in DarculaLaf .
Catching RuntimeException instead of ProcessCanceledException in JobUtil .
Added import of javax.script.ScriptContext and cleared bindings in releaseEngine method .
Added debug logging for malformed HTTP headers
Throwing IllegalArgumentException with correct expression string in ExpressionUtils .
Added &apos; touch ( ) &apos; method call to HttpContent before adding it to the received list in ContentStreamingRequestAction .
Increased line number area width by 2 pixels to accommodate longer line numbers .
Added import statement for org.jitsi.util and added method loadLoggingConfig to load logging config if present . Also , updated start method to call loadLoggingConfig after setting bundle context .
Added constructor for ORecordBytes that takes an ODatabaseRecord as parameter and sets the database context .
Fix illegal argument exception message for GraphAnalysisInjectBinding
Config.java updated : reduced max hint window to 1 hour , and updated in-memory compaction limit to 64 MB
Added test for VpTreeNode.findNearbyPoints ( ) with multiple nearest neighbors .
Added import statements for Google Guava and removed unnecessary code
Increased length of generated session IDs from 18 to % d
Added constructor for BasicDBObject and BasicBSONObject to allow for creation with a specified size
Added TerrainGrid initialization with SceneApplication.enqueue ( ) and added import statements for SceneApplication and Callable .
Added check for remaining data in writeFinal method to optimize performance .
Removed unnecessary menu items in SherpafyCustomization and updated TourViewActivity to call startTourImpl instead of startTour .
Removed unused import statement for RecoveryMode .
Utils.java : Changed &apos; exception occured &apos; to &apos; exception occurred &apos; and &apos; tab stop position &apos; to &apos; tab stop position &apos; for consistency .
Added testInsertOneToMany and testSingleQueryExecute methods to JPAModelTest
Remount external storage for changed permissions
Renamed corrupted project files to prevent data loss .
Added method like ( Expr &lt; String &gt; left , String right ) to class Grammar
Added check for read-only file type in isTextType method
Renamed YOLO suggestions to TAG suggestions in EyeemActivity .
Ensure source window is visible when showing content
Renamed &apos; MessageBarUtils &apos; and updated documentation to clarify usage .
Added OnError method to handle unexpected errors and EOF exceptions during websocket sessions .
Added field &apos; json &apos; to class InvalidJsonException and added a new constructor to initialize it .
Eliminated unnecessary CAS loop in GatedStreamSinkChannel.exit
Replace &apos; closeSessionFuture.setSuccess ( ) &apos; with &apos; closeSessionFuture.trySuccess ( ) &apos; in SpdySessionHandler.java.
Fixed issue with topic change event report message interpreting HTML chars in channel name .
DerivedStringProperty : Removed unnecessary &apos; propertyChangedInternal &apos; method and updated &apos; deriveAndSet &apos; method to call &apos; propertyChanged &apos; .
Added functionality to search chat rooms by name
Added import statement for CompressionUtils
Added &apos; identites &apos; field to ServiceDiscoveryResult class
Updated MultiValueDictionaryTest.java to reduce number of rows from 100000 to 10000 and updated MAX _ N _ VALUES to 999 .
Update ClientAuthInterceptor to use a new single thread executor
Removed unused dependency &apos; caTools &apos; in DependencyManager .
Fixed getConfiguredNetworks ( ) to return null when no networks are configured .
Added check result description to email alert subject and body
Remove unnecessary &apos; regLatch.countDown ( ) &apos; call in ClusteredEventBusTest .
Removed unused import statements in MapReduceTest.java
Added support for Exasol tables in the schema by implementing getChildType ( DBRProgressMonitor ) method and returning ExasolTableBase class .
Protected method &apos; getExecutors &apos; in &apos; ExtensibleQueryFactory &apos; class to allow subclassing .
BackStackRecord.java : Changed comparison in setNameOverride ( ) from ! source.equals ( target ) to source ! = null &amp; &amp; target ! = null
Added test for reading and checking notification list with data block of size integer
Added import statement for com.mysema.util.ReflectionUtils and updated AnnotationMapper to use ReflectionUtils.getFields instead of object.getClass ( ) .getDeclaredFields ( ) to retrieve fields .
Added support for generating changes for missing unique constraints in Oracle databases .
Added a check for null before writing a byte to the channel in ONetworkProtocolBinary.java.
Modified OStorageDistributedHazelcast.java to use a different constructor for HazelcastClient
Removed unused variable &apos; queryExtensions &apos; in PipelineRepositoryTest .
Renamed &apos; PreviewSettingsTopComponent &apos; to &apos; PreviewSettingsTopComponent &apos; and &apos; PreviewTopComponent &apos; to &apos; PreviewTopComponent &apos; .
Renamed ParseException to EvaluationException
Added InformationSchemaService and SQLParseService instances to node injector and created tables t1 , t2 , and t3 with different sharding and replication configurations .
Update TaskViewServiceTest to use TaskConfigProperty instead of Property for task configuration .
Added exclusion for ConnectivityManager sInstance field to prevent leak
Added getServerAddress method to SingleChannelMongoClient
Added support for tab and space keys in ContactListSearchKeyDispatcher .
Added some short methods for Iconics class
Fixed type caching for Python builtins .
Fixed issue with item click handling when scrolling
Remove unnecessary null check and exception handling in DTree.node ( ) method .
Renamed &apos; idGenerator &apos; to &apos; sessionIdGenerator &apos; in Manager.java and ManagerBase.java
Added implementation for open ( ) method in LocalFileSystem class .
Added static instance variable sInstance and updated systemReady ( ) method to use it .
Added check to prevent removing notifications to all users when no package name is specified .
Fixed issue with emitting response to matching observables in RxPermissions .
Added code to handle delay for first stop arrival time when updating timetable .
Added numDownloadedItems to NavDrawerData constructor
Added check for store definition mismatch between local and remote stores in VoldemortBuildAndPushJob .
Renamed &apos; compaction _ strategy &apos; to &apos; compaction &apos; in CFPropDefs.java.
Deprecate defaultReadOnly field in DataSourceFactory and replace with readOnlyByDefault
Changed image / jpg to image / jpeg and video / mp4 to video / mp4 in setDataAndType method of BaseCaptureActivity.java
Added import statement for TimeoutUtil and updated sleep time from 2 seconds to 3 seconds using TimeoutUtil.adjust ( 3 ) in StatefulTimeoutTestCase .
Added a check to set interest ops on a channel only when it &apos;s not a succeeded future .
Added try-catch block for NativeOsmandLibrary preloading to handle exceptions and improved error handling .
Increased query cache key buffer size for search queries
Added check for JSPX support in XHTML files for JavaBreakpointType .
RealmChangeListener : updated documentation and added explanation of how it works with background threads
Added mkdirs ( ) method call to create directory for csv metrics
Added BlockFlower to soft blocks
Fixed printing of copyright notice and improved formatting in DependencyInjectorGenerator .
Fixed close method to be private
Added caching for loaded plugin implementations to avoid duplicates
Added imports for AlarmCallback , AlarmCallbackService , NodeService , Stream , StreamService , User , and UserService in AlertsController.java.
Added &apos; Kilogramm &apos; and &apos; Uhr &apos; to NOUNS _ TO _ BE _ IGNORED in AgreementRule.java
Added methods to set and get slow SQL milliseconds and log slow SQL , and added a method to check if connection stack trace is enabled .
OfflineMessageStrategy : Ignore or bounce messages of type normal , groupchat , or headline based on strategy
Added check for frameDataRemaining = = 0 in AbstractFramedChannel.read ( ) method to null out receiver variable and avoid potential NPE .
Added support for JSPX files in custom search
Refactor ONetworkProtocolBinary to use server.openDatabase instead of creating a new database instance .
Update SSLContext creation to use require _ client _ auth
Added import statements for JetWithJdkAndRuntimeLightProjectDescriptor and org.jetbrains.annotations.NotNull , and updated getProjectDescriptor ( ) method to return JetWithJdkAndRuntimeLightProjectDescriptor.INSTANCE.
Added method removeWallet ( Wallet ) to AbstractBlockChain and updated listeners .
Added import statement for com.liferay.registry.Filter and modified ServiceTracker to use a filter to track services of type ControlPanelEntry .
Fix for SSL recoverable exception in SslValidatorDialog
Added abstract and interface modifiers to V3ClassAdapter and updated access flags for inner classes .
Remove unnecessary @ NoInspection annotation in PropertiesExpander constructor .
Added support for creating a new person with age in Couchbase
Added keyboard shortcuts for delete , forward , reply all , and reply buttons in MessageView activity .
Update LinkedInApi to use authenticate instead of authorize for OAuth flow
Added import statement for tachyon.thrift.FileSystemMasterService
Fix for bug # 12345 : Handle case where loadMetadataSuceeded is false
Fixed error handling in Inputs.java by removing unnecessary WebApplicationException and continuing instead .
Removed unused import statements and removed unnecessary OnTabItemClickListener implementation .
Fixed a bug in ServletClientSocketChannel.java where lock.lock ( ) was unnecessary and caused a deadlock
Added Apache License header to IFileDownloadServiceHandler.java.
Fix incorrect commit handling in InMemoryTitanGraph.java
Removed deprecated method checkResources ( DeployedApplication ) and replaced with checkResources ( DeployedApplication , boolean ) . Will be removed in Tomcat 9.0.x.
Removed unnecessary home button handling in TaskListActivity .
Increased socket timeout from 1000 to 5000
Added import statement for Java &apos;s &apos; format &apos; method and changed the return type of ModelType.java &apos;s toString ( ) method to match .
Fixed comparison in RestController.java and RestFilterChainTests.java to use Integer.compare ( ) instead of subtracting integers .
Added import statement for android.support.annotation.NonNull and modified BottomSheetHideable class to use NonNull annotation .
Added check for onPause before finishing TaskEditActivity
Added logic to fill the next message in the mutation buffer for ShardedDistributedMessageQueue .
Added import statements for BindingListener and InstanceId , and updated ClusterClient to log connection and disconnection events .
Added support for multiple mongos hosts in TreasuryYieldXMLConfig .
Added configuration file listener to GoConfigWatchList
Removed unnecessary import statements for Location and JSONObject in TestType.java.
Modified CommonUtils.java to use TachyonURI.SEPARATOR instead of &apos; / &apos; for consistency with Tachyon &apos;s path separation .
Added getLastSelected method to Selection class .
Added null check for model before unlocking and stopping training in SpeeDRF.java
Replace first occurrence of token with its uppercase version in sentence .
Update AdminConnectionVerifier to use a single connection per node
Added check for quote start and end in SQLAutoIndentStrategy
Added OnAccountDisabledListener and implemented onAccountDisabled method to clear chats for disabled accounts .
Added methods getEnergyStored ( ) , maxCapacity ( ) , and minimumConsumption ( ) to MjAPI class .
Added support for on-demand native management services and HTTP management interface in DomainModelControllerService .
Remove unused method typeKeyspace ( CQL3Type.Raw rawType ) and simplify function findFunction ( )
Added null check for mCallAudioState in getAudioState ( ) method .
Added constructor for LiquidTank with optional capacity parameter .
Increased font size of multi-line messages in MessageAdapter .
Added trim ( ) method call to remove excess whitespace from Java agent arguments .
Added hashCode ( ) and equals ( ) methods to KSMetaData class .
Update sendfileData creation in AbstractHttp11Processor
Renamed some appearance options and reordered them .
Renamed &apos; Use super method of &apos; and &apos; Change usages of &apos; to &apos; Change base method &apos; and &apos; Change usages &apos; in GrIntroduceParameterHandler .
Added support for polygon and linestring conversions in AbstractConverterTest .
Added support for multipart requests in ServingLayer
Added import for &apos; Registry &apos; and updated logic for editor rendering
Removed unnecessary &apos; WHATS _ NEW _ DIALOG &apos; case from ShowsActivity .
Added method getChildCount ( ) to class PShape .
Fixed bug where cube would not switch to follow state when space key was pressed .
Replace integer division with float division in AbsSeekBar.java
Added support for Spring Roo JPA entity annotations in QueryDSL APT processor .
Fixed issue with ActionUrl.java - now properly handles ActionGroup cases .
Enable lexing optimization correction check
Remove unnecessary assertions in CFMetaData constructor
Removed debug build enablement for apk as it &apos;s no longer working with R22 SDK .
Removed unnecessary variable &apos; TAB &apos; from Invocation.java
Modified OHazelcastDistributedDatabase class to break instead of returning request when waitForMessageId is reached .
Updated DevicePolicyManager.java to clarify that only device owner admins can set the WIPE _ RESET _ PROTECTION _ DATA flag without throwing a SecurityException .
Added method createProperty to JsonElementGenerator
Added CDI support for Camel applications
Removed unnecessary import statement for ChunkSatelliteView .
Deprecate addArtifacts method and add addTransitiveArtifacts method to prevent increased memory use .
Reinit conversation fragment and update messages .
Added support for theme changes by restarting the activity stack
Fix read method in DataSourceExtractorInput to handle interrupts and edge cases correctly
Added null check and updated string representation for OnErrorThrowable
Update RunMojo.java to use &apos; camel.usdDot &apos; instead of &apos; false &apos; as default value for DOT file generation and add &apos; camel.logClasspath &apos; and &apos; camel.useBlueprint &apos; parameters .
Disable notifications during initial setup wizard and set up notification sound
Introduce CommandContextFactory Interface
Removed unnecessary &apos; SecurityUtil.remove ( instance ) &apos; call in StandardWrapper.java
Added mOuterInterface field to Vpn class and updated check method to use it .
Added import statement for java.util.Objects ; and changed equality check for function names in CallGraph.java
Added Remy Maucherat as a co-author to CometEvent.java
Modified AbstractMessageTask.java to add &apos; getDistributedObjectType ( ) &apos; method and updated implementation to use it .
Renamed &apos; description &apos; to &apos; title &apos; in AbstractCamelController .
Replace UnknownHostException with UnresolvedAddressException in AbstractEpollChannel
Fixed incorrect type check in XmlAspectImpl .
Added digit characters to alphanumeric char array in CipherUtil .
Fixed resolution of ancestor classes in PyClassTypeImpl
Remove unused logger variable and restrictive path names .
Added import for com.google.common.base.MoreObjects in Bindings.java
Added method javaToNativeARGB for converting OpenGL format to ARGB format .
Fix for IDEA-20043 : Add missing bound check for button groups in AsmCodeGenerator
Update GitHistoryUtils.java to use final List &lt; String &gt; for LOG _ ALL and remove unnecessary private constructor .
Fix catalog loading error handling for tenant record ID
Add project key to new issue creation
Renamed CLUSTER _ PROPERTY _ NAME from &apos; cache.clustering.enabled &apos; to &apos; clustering.enabled &apos; in ClusterManager.java
Added StringLiteralQuotesAnnotator to PythonLanguage class .
Added call to myDebugger.disconnect ( ) in PyDebugProcess.stop ( ) , and added call to myDebugger.close ( ) in PyDebugProcess.processTerminated ( ProcessEvent event ) .
Added delay to allow for retry failure before clearing state and removed auto pay off to schedule a payment retry .
Added loop to setIfAbsent method in DefaultAttributeMap class
Added a line of code to increment variable i in TestEncodingDecoding.java.
Removed unused constants TYPE _ FORM , TYPE _ SUBMIT , and TYPE _ RESULT from DataForm.java
Added logging statement for testing preference in SearchPreferenceTests .
Added check for empty activity list before iterating over it in ActivityStack .
Update file store to use MinimalFileTree and SingleIncludePatternFileTree instead of DirectoryFileTree .
Fixed issue with sliding up panel layout . Changed the calculation of the top of the panel when collapsed and added a check for covered fade color .
Added a new variable &apos; partialMaskWasMatched &apos; to keep track of whether a partial mask was matched in the QTestUtil class .
Added check for non-existent package in getApplicationEnabledSetting ( ) method .
Update ThreadExecutor constants and remove unnecessary variables .
Removed unused imports and variables , simplified IBeaconDataFactory and removed NullIBeaconDataFactory .
Fixing the default priority parsing to avoid throwing a ConfigurationException and instead log a warning .
Added Apache License 2.0 header to CursorException.java.
Added setting for surround selection on quote typed
Update MySQL connection cleanup
Added RedisException handling to RBatch.execute ( ) method .
Added chatTest method to MetaBroadcasterTest class .
WindowStateAnimator : Apply translation after screen rotation animation for correct positioning .
Remove unnecessary &apos; Nonnull &apos; annotation in IssuesFinderSort constructor .
Added header manipulation for Accept and Content-Type headers in BatchOperationService
Added KeyguardViewStateManager and wired up necessary listeners / callbacks .
Update copyright year in TreeExpandableItemsHandler.java from 2013 to 2015 .
Added method getContext ( ) to FakeDialog class .
Deprecate use of AtmosphereConfig # resourcesFactory in favor of AtmosphereConfig # getBroadcasterFactory
Replace StringUtilRt with StringUtil in PyIntroduceConstantHandler .
Renamed method getstreamIdentifier ( ) to getStreamIdentifier ( ) in class SctpPayload and updated its usage in class SctpSendBufferPool .
Update comment status in SQLite for successfully moderated comments .
Replace FINE log level with INFO log level in JulLogger .
Modified DomainXml.java to use get ( ) method for accessing modelNode.get ( DEPLOYMENT ) and modelNode.get ( SYSTEM _ PROPERTIES ) instead of directly using modelNode.hasDefined ( DEPLOYMENT ) and modelNode.hasDefined ( SYSTEM _ PROPERTIES )
Added polymorphic cluster IDs support
Added test for date conversion in Message class
CameraBinderDecorator.java : Changed UnsupportedOperationException to CameraRuntimeException for unknown camera device errors .
Renamed variable &apos; mImgPattern &apos; to &apos; IMG _ PATTERN &apos; and updated regular expression pattern to include optional leading whitespace .
Added WebLogic repackaging workaround for classes in Environment.java
Fixing typos in AnalyticsTrackerNosara.java
Update atmosphere.xml to use @ _ _ _ Service instead of @ AtmosphereHandlerService for AtmosphereHandlers .
Added import statement for &apos; FailedToCreateRouteException &apos; and updated the test case &apos; HttpClientRouteTest &apos; to catch and assert the correct exception .
MavenRootModelAdapter : removed unnecessary exclusion of compiler output folders
Added openSession method and updated openForPublish to call openSession with correct permissions .
Added support for displaying notification content text for pending drafts .
Added optional parameter to loadWorld method in MixinMinecraftServer.java
Added missing &apos; public &apos; modifier to HeliosDeploymentResource.before ( ) method .
Added import statement for Arrays ; filled checkedIntermediates array with true values ; removed unnecessary line of code
Removed unnecessary BKS dependency in ServiceModule
Added command to ClearCaseRepository for lshistory
Added custom load more view for UltimateRecyclerView
Fixed issue with LwjglCanvas not being properly displayed
Removed setting for plugin loading from classpath in KuromojiIntegrationTests .
Remove @ foo and @ bar tags from Cucumber options
Update accumDots method in Omega class to use index instead of iteration .
Added methods for setting and getting repository and snapshot repository , and added method for adding protocol provider jars .
Change AtmosphereResourceImpl to cancel websocket request when timeout
Removed unused import statement for CookieGenerator .
Added random header value for relevant header in ContextAndHeaderTransportIT
Added template directory to Django project creation test .
Added cancel method to DaemonProgressIndicator , including exception handling and disposal of traceable disposable .
Fixed issue with renaming thread due to SecurityException .
Added support for regular expression based URI matching in FakeHttpLayer .
Added NetRCCredentialsProvider to GitCloneFunction to support authentication with remote repositories .
Added import statements for Logger and LoggerFactory in HikariConfig.java
Added joystick input and updated key input
Added check for CompositeElement in ASTWrapperPsiElement # getChildren
Fixed assertion in XmlElementFactoryImpl.java
Update density-dependent values in DrawableContainer
Added hashCode method to WebAuthenticationDetails class .
Added import statement for &apos; List &lt; String &gt; &apos; and added a new method &apos; userInput &apos; that takes a list of valid responses as an argument . Also updated the existing &apos; userInput &apos; method to use the new method .
Fixed command line utility to use shortenPathWithEllipsis instead of firstLast for long commands .
Added &apos; NONE &apos; category to Rule class
Added serialVersionUID to ToroTaskExecutionException
Added mandatory constraints for edges and updated the creation of vertices and edges in the SQL create edge test .
Removed LifecycleListenerRule for Engine and Host in favor of CopyParentClassLoaderRule .
Added JMX stats for percentage contended
Added implementation of OnEmojiconClickedListener and OnEmojiconBackspaceClickedListener interfaces in EmojiconGridFragment and EmojiconsFragment
Added method applyLearningRateScoreDecay to class RNTN .
Fixed logging message for OracleDatabase check compatibility mode .
Added method canScrollHorizontally ( int direction ) to AbstractChartView to enable scrolling in ViewPager .
Added lineIsNonEmpty and appendLineBreak variables to keep track of line empty status and whether to append a line break .
Update Banner.java to use UIUtil and Darcula theme support .
Added outputJunk parameter to AtmosphereFilter.outputJunk method .
Added method ResponseBody.bytes ( ) and updated method ResponseBody.string ( ) to include a warning about loading large response bodies into memory .
Added ACTION _ SHOW _ REGULATORY _ INFO intent action to Settings class .
Added import statement for Java &apos;s &apos; File &apos; class in DxStep.java
Added authentication state listener in FirebaseLoginBaseActivity .
Added ShaderVar constructor with vec3 value .
Modified Getter annotation to use &apos; lombok.AccessLevel &apos; instead of &apos; java.lang.annotation.Target &apos; for access level specification .
Update JsContext.java to replace NotImplementedException with UnsupportedOperationException .
Added test method &apos; test _ chain _ insert &apos; to SimpleDaoTest.java
Added a TODO comment regarding the use of clear ( ) in IndexedSet.java.
Fixed bug in TaskState.inDoneState ( ) method
ZoomPanLayout : updated layout method to use mScaledWidth and mScaledHeight instead of mScaledWidth - 50 and mScaledHeight - 50
Added GeometryFactory instance for building edge-based graph
Removed static final variable SHORT _ STRING _ MAX and updated ENCODER _ FACTORY initialization .
Added tripTimeSubsetCache to Graph constructor
Added method repeat ( char , int ) to StringUtils class .
Added DKV.put ( tfr ) to store the trained model in the DeepLearningTest class .
Removed unnecessary import statement for &apos; org.junit.Before &apos; in ApplicationInjectedActivityTest.java
Added display metrics calculation for remote image loading in RemoteImageLoaderTextViewAdapter .
Added SLF4J logging to BufferPoolMetricSet and updated JMX attribute gauges to handle Java 7 + environments .
Ignore MalformedURLException in TagFileProcessor when tagFileJarUrl is null
Fixed PubMulticastTest.java to use a new Aeron.ClientContext instance for each test .
Replace &apos; com.google.common.base.Throwables.propagate &apos; with &apos; io.airlift.units.DataSize.Unit.MEGABYTE &apos; and &apos; throw Throwables.propagate ( e ) &apos; instead of &apos; throw propagate ( e ) &apos;
Update PreferenceScreen.java to conditionally show title bar based on title availability .
Added link to Hibernate documentation for executing HQL queries
JUnit38ClassRunnerTest : Add RejectAllFilter to skip tests
Change equality check from using raw integer values to using getX ( ) , getY ( ) , and getZ ( ) methods .
Update FoldingCell.java : Add javadoc comment and fix illegal argument exception to illegal state exception .
Added setMediaFile method to WPEditImageSpan to set media file for images with custom sizes .
Added commit method call to init method in Neo4j2Graph class .
Added support for query string as request body in AtmosphereFramework .
Added claimStrategy.setSequence ( sequence ) in Sequencer.forcePublish ( ) method .
Added check for zero-length annotations in AnnotatedElementUtils .
Replace Windows path delimiters with Unix-style delimiters in IndexDatabase.java to avoid conflicts with Lucene escape character .
Added Apache Commons Lang import and fixed remote blog ID parsing for self-hosted blogs .
DEPRECATED - Use org.apache.camel.test.spring.CamelSpringDelegatingTestContextLoader instead of JavaConfigContextLoader .
Remove unused variable &apos; lz4Fast &apos; in CompressedObjectStrategy
Added import statement for PyUserSkeletonsUtil and skipped user skeletons directory in PyTreeStructureProvider .
Update HystrixCommandTestWithCustomConcurrencyStrategy.java
Renamed variable ALL _ POSSIBLE _ FIELDS to allPossibleFields in SearchAction.java
Reverted change to clear application user data temporarily
Update ZIP compression level range from 1-9 to 0-9 in ODatabaseRecordWrapperAbstract and ODatabaseWrapperAbstract .
Added variable &apos; pointerIndex &apos; to method determineDrag ( MotionEvent ev ) and updated its value from &apos; getPointerIndex ( ev , activePointerId ) &apos; to &apos; getPointerIndex ( ev , pointerIndex ) &apos;
Added support for updating channel type of input and output gates in ExecutionGraph .
Added attributes to test suite element in JUnit report
4395
Added support for Probe lights in MultiPassLightingLogic and SinglePassLightingLogic .
Added test cases for supporting expecting status code when authentication error and throwing nice error message when illegal path
Added error handling for reading forgeChunkLoading.cfg file and configured dormant chunk cache size to 0
Fixing tests for Predicates.in ( String , Object &#91; &#93; , int , String ) method .
Added SortActivity class with sort feature showcase .
Added getCommandResult ( ) method to CommandFailureException and WriteConcernException classes .
Added field &apos; mContext &apos; to WebRenderer and WebViewRenderer , and removed unnecessary import statements .
Update license year in DualView.java and modify copySelection method to be static .
Added space before closing parenthesis in TestBackupToolHa.java
Added JUnit and Mockito imports for testing .
Modified ForwardingFuture and ForwardingListenableFuture to use Future &lt; ? extends V &gt; instead of Future &lt; V &gt; for delegate ( ) methods .
Fixed bug in PackageManagerService.java where downgraded packages were not properly deleting code and resources .
Added import statements for SharedMetricRegistries and SharedHealthCheckRegistries
Added super call in BlockMiningWell.onBlockPlacedBy to properly set block metadata
Added support for ripple effect on dismiss and move task buttons in TaskViewHeader .
Added null check for parser annotation
Removed unnecessary accessibility settings .
Update JcrEndpoint.java to use empty string for password instead of null
Update AjpServerRequestConduit.java to fix bug in handling of remaining bytes .
Added check for retry logic in commitBlock method of BlockMaster class
Updated the application startup message to include the base URI and root path .
Added tests.bwc.version to ReproduceInfoPrinter &apos;s appendESProperties
Fixed issue with panel hiding
Removed null check for last tile in TileAdapter
Fix for broken encrypted private key test .
Update ReflectionAllocator to use Apache Harmony &apos;s ByteBuffer allocateDirect method instead of BufferUtils .
Renamed JavaScript interface from &apos; Device &apos; to &apos; DroidGap &apos; in DroidGap.java.
Update MasterWorkerInfo to setLastUpdatedTimeMs instead of updateLastUpdatedTimeMs
Added sendAlertToPebble ( ) call in TTSCommandPlayerImpl .
Added support for percent layouts in PercentFrameLayout and PercentRelativeLayout by adding a getPercentLayoutInfo ( ) method that returns a new PercentLayoutHelper.PercentLayoutInfo instance .
AudioManager.java : Changed method signature from &apos; set &apos; to &apos; get &apos; .
Added thread name with version information
Added test cases for getResource ( ) method .
Added null check for injectable objects in InjectableObjectFactory
Fixed bug in CodeSmellDetectorImpl : removed unnecessary loop and fixed exception handling .
Added sorting logic for primary blog in SitePickerAdapter .
Changed surround selection option from &apos; Quotes + Brackets &apos; to &apos; Quotes &amp; Brackets &apos; in EditingPreferencesPane and UIPrefsAccessor .
Update user agent string to Chrome desktop version if not already set
Fixed typo in FileOperationsHelper.java
Added import statement for UriUtil and added code to initialize fieldByName map in MappedContainerTypeHandler constructor .
Update JvmLibrarySpec to use component-level dependencies instead of API-level dependencies .
Replace ECDH with DHE in CipherSuiteConverter handshake algorithm .
Added public constructor to Decal class
Fix incorrect class comparison in GDTabActivity .
Fix for dispatchMenuItemSelected ( mMenu.getRootMenu ( ) , this ) - &gt; dispatchMenuItemSelected ( mMenu , this )
Added getModel ( ) method to NotificationComponent and modified add ( ) method of IdeGlassPaneEx to use NotificationsListPanelWrapper
Added dependency on JUnitTestSuiteBinarySpec for JavaPlatform
Fix JSON parsing issue in WxMpServiceImpl.java
Set all changed flags in AbstractRecord when one value is changed
Changed default browser path for Linux from null to $ { myLinuxPath }
Fixes issue with Recents not being dismissed when PIP moves to fullscreen . Adds call to dismissRecentsToLaunchTargetTaskOrHome ( ) in onMoveToFullscreen ( ) .
Update RuntimeEnvironment.java to use NumCacheRenamedThreads instead of NumCacheThreads for history renamed executor .
Added stone and cobblestone to OreDictionary
Fixed a bug in the tourToString method by initializing the string variable &apos; t &apos; to an empty string instead of a new String ( ) .
Added null checks for menuInfo and its parent view in onCreateOptionsMenu
Restore power mode after DHCP operation
Update CardListView to use Log.w instead of Log.e for a more appropriate warning message when using a generic adapter .
Added support for mangling method names to avoid collisions in the generated JavaScript code .
Replace constant impl with generic constant impl for method parameters .
Removed unused import statement for DeltaManager
Removed unnecessary log statements in ConnectPlugin.java
Added support for XML data type in OracleTable.java
Added check for pre-layout state before dispatching child selected event .
Added handling of HTTP status codes SC _ FORBIDDEN in EduAdaptiveStepicConnector .
Fix for fault tolerance test . Remove unnecessary sleep and add new test for multi-master setup .
Added support for season watched notifications
Removed unnecessary log statement in CompileDriver.java
Fix for IllegalArgumentException in FLogDefaultLoggingDelegate.isLoggable
Change in PersistentHashMap2.java : modify HashCollisionNode construction
Added plugin settings and updated plugin meta manager to use them .
Added contains ( ) method to FileUtils class to test if a file lives under a given directory .
Fix exception handling in TieredBlockStore # commitTempBlockMeta
Fix request / runner cycle
Added checkArgument for bufferMaxID and bufferNextID in StandardIDPool
Increased performance test timeout for decompiling JTable.class from 2500 to 4000
Added support for new keywords : too and inn .
GH-992 : Implemented adding a new account
Move print statement from System.err to console.getStdErr ( ) in AbstractCommandRunner
Removed import of Log and Constants , updated package name in OkHttpRestClient.java.
Remove unnecessary FragmentTransaction usage and add a note about avoiding mixing android.app.FragmentTransaction with support Fragment .
Added EmittingLogger to CachingClusteredClient and replaced Logger with EmittingLogger in the constructor . Also , updated the log statement in the method getDruidServer to use the new EmittingLogger .
Added consistency check for presence objects with equal status and from addresses .
Added import statements for Exchange and AggregationStrategy in SplitWithEndTest.java and SplitWithNestedFilterShouldSkipFilteredExchanges.java
SplitProcessor : Changed splitting logic from using ArrayList to using String &#91; &#93; for better performance .
Added aggregator with header &apos; cheese &apos; in AggregatorTest .
Removed unnecessary method call to &apos; cleanupRepository ( ) &apos; in JpaRouteTest .
Added documentation for HTTP response codes and possible error types in AjaxStatus class .
Added support for ExpandableListActivity in ListAdapterWithProgress .
Added GSP _ FILE _ EXTENSION and GSP _ TO _ RENDER constants to GrailsApplicationAttributes
Remove unnecessary &apos; throw error ; &apos; statement in HazelcastTestSupport.java.
Fixed issue with alpha slider not updating properly when color is null .
Added getVariantForTrip method to TransitIndex class
Added support for adding many test processes in background
Renamed variable &apos; metricid &apos; to &apos; metric _ id &apos; in class IncomingDataPoints .
Added logging for legacy route addition with UID / PID for ConnectivityService
Rename Util : updated doRename method to use RenamePsiElementProcessor.forElement ( element ) .getPostRenameCallback ( element , newName , elementListener ) instead of elementListener.getElementRemoved ( ) for better handling of post-rename callbacks .
Added check for empty list in getFirstVisiblePosition ( )
Removed unnecessary setToolTipText call in FileHistoryPanelImpl .
Ensure that comet connections are not closed on timeout .
Added &amp; to home address field in VCardTest .
Added method ensureDirectoryExists ( File dir ) to class FS and modified ModulePersistence to use it .
Added wildcard character to contactJid query to match conversations with multiple contacts
Added check for EOF while reading raw store records .
Added code to handle renaming of variables in ASTGenerator .
Added support for specifying encoding when creating SMPP messages
Added name attribute to DeviceOwner serialization
Removed unnecessary socket timeout setting in NioEndpoint
Fix copy array in PackageManagerService
Added warning about non-determinism in Test.java
Fix PyTypeCheckerInspection to correctly handle iterable types .
Removed check for isReadyForRead ( ) in WsFrameServer.onDataAvailable
Modified PhysicsSpace.java to use a different collision detection approach .
Added support for fallback key events in ViewRootImpl
Removed unnecessary pipeline.fireChannelActive ( ) call in AioSocketChannel.completed0 ( )
Updated example value of quality profile key from &apos; java : Sonar way &apos; to &apos; sonar-way-cs-12345 &apos; in SearchAction.java.
Update lookup method for StatefulBean in StatefulServlet
Added annotations for constructors : AllArgsConstructor , NoArgsConstructor , RequiredArgsConstructor
Update button enabled when update is available and list is downloaded .
Added import statements for BytesStreamOutput and CachedStreamOutput , and updated copyToByteArray method to use CachedStreamOutput for better performance .
Add constant and linear functions to technical debt model
Added new transaction type for raise fee and updated ASSURANCE _ CONTRACT _ STUB to include RAISE _ FEE .
Added a new editor for the settings of this extension .
Fixed bug in MethodDefinition.java : now properly quotes parameter names when writing them out
Fixed exception type in NioDatagramWorker from NotYetConnectedException to NotYetBoundException .
Added check for HTTP response code 101 in HttpMessageDecoder
Added simple field resolution implementation and updated test class .
Fixed typo in LocalizedString for unnamed street .
Fixed bug in DefaultBroadcaster where cache was not being properly checked .
Added support for DEFAULT modifier in SuggestedFixes
Added @ Property annotation to isPlaying method in OALAudioTrack.java
Fix bug in IndicesRequestTests : assertAllRequestsHaveBeenConsumed now uses emptyIterable instead of isEmpty
Added methods onLayoutScrapList and setOrientation to TWBaseLayoutManager to save and restore lanes .
Hacky fix for SearchView autocomplete threshold
CompilationChecker.java : Changed ignoreOptionalProblems ( ) from true to false .
Add check for home activity before setting focused activity
Factor out voice search settings
Added logic to guess number of columns in CSV file based on data
Optimized GlyphPage.java by reducing texture dimensions to match the actual size of the glyphs .
Added null check for resource loading in JavaPlugin
Added method setIcon ( Icon , int , int ) to LayeredIcon class
Fixed execution error handling in AsyncGenericProgramRunner
Added test for handling protocol-relative URLs and CDATA
commit message
NimbusClient : updated constructor to take integer timeout value instead of string
Fixed bug in NettyAsyncResponse.java : status check now uses status variable instead of body parts
Fix scope chooser combo to use PSI _ ELEMENT instead of PSI _ FILE for data context element .
Added method hasFeatures ( ) to InterpreterTest.java
Added method getLastItem ( Collection &lt; T &gt; ) to class ContainerUtil , and modified subtract ( Collection &lt; T &gt; , Collection &lt; T &gt; ) method to use HashSet .
Added &apos; shuffle _ training _ data &apos; to the list of experiment flags in DeepLearning.java.
Updated FileStaticAssetRequestHandler to use Ratpack &apos;s clientError method instead of manually setting the response status and reason phrase .
Fixed bug in Validator.java where test result didn &apos;t agree with control result on first try .
Added SocketIO client abstraction and made server thread-safe
Added toString ( ) method to AttachableImpl class .
Remove @ SuppressWarnings ( &quot; deprecation &quot; ) annotation from DataFormatDefinition.getDataFormat method .
Added ability to read activity log entries from a string buffer
Exclude null and separator actions from GotoActionModel
Modified UserConfig class to use &apos; isPreventStandby &apos; instead of &apos; preventStandbyEnabled &apos; and added &apos; setPreventStandby &apos; method . Also , updated ActionFileHandler to use &apos; isPreventStandby &apos; instead of &apos; preventStandbyEnabled &apos; .
Update copyright year in GitCherryPickAction.java from 2012 to 2014 .
Added check for full screen player on first time only in MusicPlayerActivity
Updated Scroller.java and Solo.java to use consistent naming conventions and improved code readability .
Update CharacterControl.java to remove deprecation and add suggestion to use BetterCharacterControl .
Added TreeNode node to getChildAfter ( groupNode ) in TreeContactList.java
Removed unnecessary @ SuppressWarnings annotation in BasicInvocation.java
Fixed bug in isClassFiltered method in DisUnitySettings.java
Added try-catch block for handling exceptions during cluster state publication .
Added check for delegate being an ExpressionHandler before setting filter in TopLevelMatchingHandler constructor .
Removed unnecessary cast and fixed equality check for IndexableSetContributor
Added check for project disposal before running compiler
Changed color filter mode from SRC _ ATOP to MULTIPLY in QKLinearLayout .
Removed unnecessary &apos; name &apos; property from @ Example annotation .
Fixed background color issue in DarculaCheckBoxUI .
Fixed issue with executeOnPooledThread method in ApplicationImpl class .
Fixed reference to UserDetails interface in UserDetails.java
Fix for C + + editing target : added check for dirty state before saving and executing source command .
Added check for Theme.AppCompat in RenderSessionImpl
Added logging for caching restrictions until expiry date
Added shutdown method call in onDestroy for WeatherDemo and WeatherPlusService to properly close connections
Added line break to read string in WordVectorSerializer
Added test for episode cleanup num days
Modified OsgiSpringCamelContext to use &apos; this &apos; instead of &apos; ctx &apos; for the bundle context
Added public modifier to backgroundSymbol field in ConstantsAndVariables.java
Added ToolwindowTitle action place
Renamed variable &apos; followingAccount &apos; from &apos; String &apos; to &apos; protected String &apos; .
Update AhcBridgeEndpointTest to use application / xml content type
Removed unnecessary import statement for &apos; PatternRule &apos; and simplified code for creating checkboxes in ConfigurationDialog .
Added altitude to PokemonGo constructor and LocationFixes class
Added support for NOT predicate in StatementAnalyzer .
Update license year in OChannelBinary.java
Modified isHasMoreData ( ) method in ResultSetViewer to check execution context before calling dataReceiver.isHasMoreData ( )
Update clearPersistedFilesTest to use different block IDs .
Remove static Wire instance and use shared Wire instance instead .
Added check for move or copy capability before performing action
Removed unused method dumpPerformanceMetrics and related code .
Fixed race condition in publishPresenceStatus by adding null check .
Added a try-finally block in send ( ) method to release wake lock on errors
Added lowercase workaround for QualifiedName in SymbolAllocator and added tests for wildcard from subquery and case-insensitive output alias in order by .
Fixed syntax error in BindingClass.java and updated OnTextChangedTest.java to use view1TextWatcher instead of this.view1TextWatcher.
Removed unused constructor and commented out code in WifiConfiguration.java
Added functionality to create courses directory and cache file if they don &apos;t exist
Modified NiftyJmeDisplay.java to use InputSystemJme with inputManager parameter .
Rename Sybase ASE to Adaptive Server Enterprise in DbSupportFactory
Added null check for servlet registration
Added @ AwaitsFix annotation to testOldIndexes method in OldIndexBackwardsCompatibilityIT.java to track issue # 17514 .
Added logging of incoming packet version and sender address in MulticastService
Added support for using Groovy &apos;s @ Grapes annotation with Java 8 &apos;s @ FunctionalInterface
Fixed bug in XPub.java : Unsubscribe is not verbose .
Added login options bundle for custom tokens
Remove unnecessary String comparison and close client after test .
Add &apos; contacts &apos; as an option to java _ test ( )
Added restriction for user agent string in XWebView.java
Remove disabled tests for getBackgroundColor and setBackgroundResource
Fixed a bug in WifiManager.acquire ( ) where mRefCount was not being properly incremented when acquiring a multicast lock .
Update to use container.getCatalinaBase ( ) instead of Bootstrap.getCatalinaBase ( )
Disable debug mode for MultiWaveView
Added imports for Android support annotations and updated sanitizeSiteUrl method to use trim ( ) instead of substring ( ) for removing padding whitespace .
Update generated ClusterSummary.java from Thrift Compiler ( 0.9.2 ) on 2015-3-2
Added check for existing temp directory and handling for non-directory temp location
Removed unused imports in PlayWorkerServer.java
Added @ Slow and @ TestLogging annotations to SearchWithRandomExceptionsTests # testRandomDirectoryIOExceptions
Fixed bug in Dashboard constructor by setting &apos; global &apos; field to false by default .
Added volatile fields for maximumBinaryBufferSize and maximumTextBufferSize to UndertowSession.java
Added method getCoyoteRequest ( ) to class TesterDigestAuthenticatorPerformance .
Added constructors for RateOptions with counter and counter _ max parameters .
Update StatsGeoviewsFragment.java to use double quotes for string literals in Google Charts initialization .
Added slash at the end of WEBJARS _ LOCATION constant .
Removed unnecessary import statement for ResValue .
Added import statement for VfsUtil and updated copyAsFiles method to handle null defaultTargetDirectory
Added error message when loading font fails .
Replace Striped.lock ( ) with Striped.lazyWeakLock ( ) in FeedRefreshUpdater
Removed unnecessary constructor from TiledLayer class .
Update copyright year in CompletionUtil.java from 2009 to 2015 and add check for empty prefix in sortMatching method .
Fixed issue with wheelchair accessible transit links .
TCPNetworkConnection : filter out client mode nodes with zero height
Added -classpath option to JetTestUtils.compileJavaFiles ( ) for KotlinDebuggerTestCase
Set deferred locking to true by default
Added method isSubclassOf ( Class ) to KeySnapshot and updated FramesHandler to use it .
Removed unnecessary source root for kunit library in StdlibTest .
Added logic to allocate payload type for formats that don &apos;t have one yet .
Modified AtmosphereServlet.java to use declared constructor for CometSupport instance creation
Fix BridgeMethodCompiler to pass correct argument count to target function
Added test case for shadowing previous result in evaluator
Added import statement for java.util.Locale in MultiPartInputStream.java.
Added logger and debug logging for AdvancedLeakAwareByteBuf
Update store version in Busted format from vE.B.0 to vC.B.0.
Added test for OperatorAllTest.testFollowingFirst
Disable hard consistency checks in PersistentHashMap
SyncScrollSupport : Changed setDisabled method to return void and added a new visibleAreaChanged method .
Increased sleep time from 50 to 100 milliseconds in DialogUtils .
Added null check for list in MemorySection # getList
Modified AvailableJavaHomes to return non-current Java versions
Added hashcode to backup path for better uniqueness
Add null check for closeable resource in RealtimePlumber # close ( ) method .
Added check for dynamic manifests in DashChunkSource
Modified PyDocstringGenerator to use myDocStringExpression.getTextOffset ( ) instead of myDocStringOwner.getTextOffset ( ) + myDocStringOwner.getTextLength ( )
ZAP : 2012 / 03 / 15 Changed the method getDisplay to use the class StringBuilder instead of StringBuffer . ZAP : 2012 / 04 / 23 Added @ Override annotation to the appropriate method . ZAP : 2012 / 05 / 28 Added some JavaDoc
Update MimeMessageHelper to use ENC _ QUOTED _ PRINTABLE for text parts instead of ENC _ 8BIT
Added import statement for &apos; SystemInfo &apos; and updated isAutoExpandNode method to consider Windows root element
GenerateDocBookMojo.java : Fixing entity declaration for XML files .
Added import statement for END _ ELEMENT and updated WebservicesSubsystemParser to implement XMLElementReader interface .
Added VfsUtil.markDirtyAndRefresh ( ) call to EditorTabbedContainer to properly refresh files after syncing .
Added DisplayInformationComponent import to BehaviorSystem and Interpreter
Removed unnecessary metaClassRegistry.setMetaClassCreationHandle ( ) call in tearDown ( ) method of AbstractGrailsHibernateTests .
Added support for screen blending mode in OGLESShaderRenderer .
Fixed issue with unique property key in TitanGraphTest .
Notify dataset invalidated when resetting map info controls .
Removed @ Optional annotation from Media.copyright field .
Added support for setRemoveOnCancelPolicy in SockJsThreadPoolTaskScheduler .
Added check for non-empty content before writing XML document in ExportHTMLAction .
Added &apos; private &apos; modifier to reset method in HttpConnection class .
Added logging statement to DefaultDaemonConnector and implemented toString ( ) method in DaemonCompatibilitySpec .
Added equals and hashCode methods to ActivityLifecycleCallbacksWrapper to properly implement wrapper behavior .
Added line to updatePropertiesPanel ( ) to request focus when hiding properties
Check ini files for download possibilities
Temporarily fix for OC-8606 by returning unmodifiable map of parent environment
Added hostname and max _ file _ descriptors details to IndexerClusterCheckerThread notification
Added functionality to navigate to a destination point using the context menu
Removed unused import &apos; com.sun.tools.javac.code.Flags &apos; in OperationEntry.java and removed unused import &apos; com.sun.xml.internal.txw2.TXW &apos; in ProxyCommitRollbackTask.java.
Added nightly test for issue 1085 : Write behind backup
Added check for lastStartTag nullity in isAppropriateEndTagToken ( )
Added @ SuppressWarnings ( &quot; deprecation &quot; ) annotation to RequestFacade and ResponseFacade classes .
JspHelper.java : Changed escapeXml method to use toString ( ) instead of String.valueOf ( )
Fixed issue with ReverseNestedAggregator &apos;s parentDoc retrieval
Added checks for unexpected tokens in HighlightBuilder .
Update Publication class to use clientConductor.isPublicationConnected ( ) instead of isConnected ( ) for checking publication connectivity .
Added a new method &apos; findById &apos; to the &apos; ButterKnife &apos; class that allows for easier injection of views from a Dialog .
Fixed issue with DocumentationManager not showing hints in certain cases .
Added import statement for SyncFlags in TagDataService.java and added transitory flag to suppress outstanding entries in ActFmSyncService.JsonHelper.featuredListFromJson
Added support for single and double quotes in CHAR _ MAP .
Added PingPongLatencyTest with queue-based and disruptor-based implementations .
Added netFilter = RegisterMode.AUTHORITY to @ ReceiveEvent annotation in CharacterSystem.java
Removed unnecessary assert statement in HttpMessageConvertersTests .
Removed unnecessary import &apos; org.junit.rules.TestRule &apos; in ClassLoadingStrategyForBootstrapInjectionTest.java.
Fixed synchronization issue in Library.initialize method .
Added imports for AuthScope and UsernamePasswordCredentials in AsyncHttpClient.java
GLMModel.java : Added validation for alpha parameter and fixed error message for p-values computation with IRLSM solver .
Added method parseMapObject ( MapObject mo , Entity e , long lid ) and called it in parseMapObject ( mo , e )
Added support for CVS roots with spaces in the configuration file .
Added missing checks for null before returning data from CommentTable and PeopleTable queries .
Added null check for second argument in SquidUtilities.addAll method .
Fixed issue with ExpandableView getting clipped by parent when translating upwards
Remove deprecated constant USE _ SUBJECT _ CREDS _ ONLY _ PROPERTY
Removed limit from Lists.Get request
Removed unused variable sender2 and logger statement in ChatSystem .
Added check for indices.getNumIndices ( ) &gt; 0 before drawing elements
Added methods to check for specific items and materials in inventory
Added &apos; private &apos; modifier to class field &apos; ch &apos; in SingleBlockingChannelEventLoop.java
Added test for project creation with appropriate column and row count .
Updated ObjcRuleClasses.java to substitute $ ( CFBundleIdentifier ) and $ ( AppIdentifierPrefix ) variables in entitlements attribute .
Added support for TYPE _ INPUT _ METHOD windows in ViewRootImpl .
Renamed indexer.searches.timeranges packages to plugin.indexer.searches.timeranges.
Removed unused import : org.terasology.module.sandbox.PermissionProvider
Update GremlinPluginFunctionalTest to use _ ( ) .out ( ) instead of out ( ) , and add autoIndexNodes = true to @ Graph annotation
Modified GeocoderResource.java to use JTS Envelope instead of Sun Jersey InjectParam for bounding box parameter
Renamed CamelCxfTest to CamelCmisTest and updated component name accordingly .
Fix for singular service handling in ActiveServices
Added import for SystemClock and updated Clock interface to use SystemClock implementation .
Added @ Ignore annotation to DropwizardApacheConnectorTest due to consistent network timeout failures on travis CI .
Update to MethodMetaInfo.java and Type.java files .
Vibrator.java : Changed vibrate method to take milliseconds instead of long , and added documentation for vibrate method
Added support for converting List and Collection to Map
Remove unnecessary @ SuppressWarnings annotation in ModelCreators.withProjection method .
Added support for parsing subsystem declarations with multiple updates .
Added error message propagation in Indexer.propagateFailure
Added check for ApplicationManager.getApplication ( ) .isUnitTestMode ( ) in SMTestRunnerResultsForm
Added method getCurrentFrameIndex ( ) and changed getCurrentFrame ( ) to return the frame index instead of the tile .
Added support for rotating luminance source images by 90 degrees counterclockwise .
Renamed reportListFile to reportLostFile in LineageMaster and updated client service handler to use the new method .
Added PublicApi annotation to SimpleRawTable class .
Remove unnecessary URL parameter from DatabaseConfiguration
Added check to not reschedule sync operation if it retried too many times .
Added nullability annotations to TestStatusListener methods
Updated DB.java to change collection size from int to int ( in bytes )
Modified BaseRunConfigurationAction.java to use configuration.getIcon ( ) instead of configurationType.getIcon ( )
Modified UserDetails interface to remove null password check .
Added support for HXX files in CAnalyzerFactory and updated AnalyzerGuruTest to include rfe2969 test case .
Set last full sync time to now and reset merged movies flag in ConnectTraktTask .
Modified DroidGap.java to handle back button presses in edit text fields .
Added serialVersionUID to EntityIndex for serialization .
Fixed syntax in ViewGroup.java to use android.view.View.AccessibilityDelegate instead of AccessibilityDelegate
Added new metrics for total started threads and peak thread count
Added type evaluation context to PyParameterInfoHandler # analyzeCall
Modified FileComponentTest to use &apos; file : myfile &apos; instead of &apos; file : / myfile &apos; .
Removed unnecessary imports and simplified ProxyHandlerSSLTest class .
Added @ SuppressWarnings annotation to onDeleteAll and onUpdateAll methods in DeleteHandler and UpdateHandler classes .
Allow disk reads in GlideExecutor thread policy to fix issue # 1170
Removed unused property PROPERTY _ IMS _ REG _ REQUIRED from TelephonyProperties .
FavoritesTreeFragment : updated search view to use FavoritesActivity &apos;s updateSearchView method
Update GrailsDomainBinder.java to change cascade setting for many-to-many relationships from &apos; all &apos; to &apos; save-update &apos; .
Added null checks for username and password in login method of HttpServletRequestImpl .
Added default title ( ZAP application name ) to AbstractFrame
Improved error logging in NioClientManager
Removed unused variable &apos; refreshRequested &apos; in RenderTaskListGenerator .
Added test for fresh receive key creation time
Update signal strength levels for LTE networks .
Added systemReady ( ) call for Lock Settings Service and Window Manager Service in ServerThread .
Added static variable &apos; transactionManagerBeanNames &apos; to ChainedTransactionManagerPostProcessor and updated constructor to initialize it .
Rename title from &apos; Unversioned files &apos; to &apos; Unversioned Files &apos; in UnversionedViewDialog .
Added setting for allow _ store _ upgrade and updated path variable .
Added Oleksandr Tyshkovets as a co-author and added a new constructor that allows appending a specified text to the result string .
Added shutdown check in ThreadProcessor to throw IllegalStateException when not running .
Added user ID check for URI permission in ContentProvider
Added empty string argument to SpanishPronounDisambiguator.disambiguatePersonalPronoun ( ) for improved disambiguation
Force creation of ActivityResourcesStruct in ResourcesManager .
Removed unused import statement for ProjectViewPane
Added trimming of index type before adding it to the list in Select.java
Added dismissKeyguard flag to PhoneWindowManager.java and updated the logic for hiding apps when the keyguard is secure .
Disabled basic authentication and set user password to &apos; password &apos; for demo purposes .
Fixed JmsMessage to correctly extract body from JMS message when using JmsExchange
Added method projectClosed to class SSBasedInspection to clear compiled configurations on project close .
Added mock ID for client in NetworkOwnershipTest .
Update FileAppenderFactory.java to change archived file count constraint from &#91; 1 , 50 &#93; to &#91; 0 , 50 &#93; and remove unnecessary imports .
Update JSONPrettyFormatterTest to handle different property orders in JSON output .
Added DELETE _ USER _ FLAGS and DELETE _ USER _ PROPS statements to DefaultUserProvider.java
Update copyright year in MotionEvent.java from 2012 to 2016 and set lookAt vector to Vector3f.ZERO
Notify old chat closed
Added &apos; ready &apos; method to WebApplication and called it in Madvoc.java
Added support for serializing boolean expressions in SQL .
Added support for option values in TaskClassValidator .
Fix ClosureCodeRemoval to correctly replace function calls with their first argument .
Remove unused methods getProtocol and setProtocol , and reduce socketCloseDelay from 300000 to 0 .
Added support for ASE and Adaptive Server Enterprise in DbSupportFactory .
Modified TestModelControllerService.java to add synchronization for content and removed unused variable remoteFileRepository .
Added import statement for FileDocumentManager and added call to saveAllDocuments ( ) before checking
Modified WindowManagerService.java and WindowStateAnimator.java to update surface boundaries for recovering memory .
Added double quote in HTTP2 _ UPGRADE _ ACK
Fixed MasterPasswordInputDialog dispose dialog instead of setting visible to false
Added tap event handling to FlickScrollPane .
Added support for finding parent View holder in AdapterUtils
Added super.contextInitialized ( sce ) in TestWsWebSocketContainer and TesterEchoServer to fix a warning .
Changed visibility of UserService.afterLogin ( ) from protected to package private to enable spying on this method .
Added synchronized method &apos; stopped &apos; to SipWakeupTimer and made onReceive method synchronized .
Update ShadedJar.java to retain only reachable classes from keep package hierarchies
Fixed issue with EnterHandler not properly handling nested comments .
Replace Promise.fromTry with Future.fromTry and fix type parameter for reduce method
Added functionality to add all query parameters from a request to a new query parameter map , instead of overwriting existing values .
Fixed VCS refresh issue by removing unnecessary refresh calls .
Added aggregator with header &apos; cheese &apos; in AggregatorTest .
Added methods retriesExhaustedLogLevel and retryAttemptedLogLevel to DeadLetterChannelBuilder
Remove unnecessary VisibleForTesting annotation and setter method for processAnnotations .
Added debug logging for adding cached messages in EventCacheBroadcasterCache
Removed unused interfaces MapTypes and ResponseTypes
LZFCompressor : Removed unnecessary logger statement .
Added deadlock detection and fixed releaseWriteLock ( ) method calls in RWLockTest .
Added try-catch block for SchedulerException and IOException in updateNote method .
Added parameter &apos; statementsToExecute &apos; to PydevConsoleRunner constructor and updated getMappings method to use it .
Added check for iOS environment variables in SystemTest .
Added support for getter methods starting with &apos; is &apos; for properties of type boolean .
Fixed bug in KdTree.java : correctly set lesser and greater nodes for non-empty lists
Escape non-alphanumeric characters in ticket names
Update Show installation error message in CreateVirtualEnvDialog
Simplify Boolean Expression Check : Added check for overly complicated boolean expressions .
Added shutdown method to Bukkit and Server classes .
Format string in BaseDirFinder.toString ( ) method .
Added import statement for android.support.annotation.NonNull and updated forceSelfHostedMode method to use @ NonNull annotation .
Update default column count for StaggeredGridView
Added synchronized block in pushFromBacklog method to avoid concurrent modification of notifications stack .
Added support for redirecting to a specific URL when an invalid session ID is requested .
Set QueryCacheConfigReadOnly.setCoalesce ( ) method to throw UnsupportedOperationException .
Removed unnecessary import of LoginException
Added code to fetch latest scan results when frequency band is set
Increased maximum height of chart images from 300 to 500 pixels and adjusted width accordingly .
Added test cases for scalar subqueries in SELECT statements
Modified Job.java to change start method visibility from public to protected .
Added JsonIgnore annotation to MessageSummary.getRawMessage ( ) method to exclude it from JSON serialization .
Added check for null &apos; from &apos; attribute in IQ packets before replying to an IQ error .
Updated AbstractTicketRegistry to use final variables for ticketStream and logoutManager in cleanupTickets method .
Increased side margin for TaskPositioner from 5 to 100 dip
Updated TelecommConstants.java to use correct constant name for call ID extra
Update copyright year in ContentsUtil.java from 2009 to 2014 .
Fixed issue with lost pointer 2 by clearing it when not needed .
Added test methods for configuration and data types in InMemoryGraphTest .
Remove @ Before and @ After annotations , add @ Rule for TemporaryFolder , and ignore flaky test method processCancelBuildCommandBuild .
Replace sliceAddresses &#91; rightPosition &#93; with slices &#91; ( ( int ) ( rightSliceAddress &gt; &gt; 32 ) ) &#93;
Renamed parameters from &apos; o &apos; to &apos; obj &apos; and &apos; millis &apos; to &apos; timeout &apos; in Object.java
Renamed &apos; BlockName &apos; to &apos; plugins &apos; in DefaultScriptPluginFactory and PluginHandler .
Update OAuthSignatureCalculator to use setHeader instead of addHeader
Added logging for unable to instantiate message in ActFmSyncThread
Fixed assert statement in Neurons.java
Added implementation for ImmediateAuthenticationMechanismFactory to return an authentication mechanism instance .
Removed unnecessary @ TargetApi annotation in GallerySettingsActivity .
Added cluster configuration entry permissions .
Update StreetTraversalPermission for footway and bicycle
Update CountryDetectorServiceTest to use systemRunning ( ) instead of systemReady ( )
Added support for suppressing inspections for specific languages
Fixed partition replica address update for migration
Replace HashMap with LinkedHashMap in Bug _ for _ smoothrat8 test
Changed blur radius from integer to float
Added static methods for creating empty HashArrayMappedTrie and RedBlackTree instances .
Update package-info.java to include guice-persist-3.0.jar in the requirements .
Fixed logging level for transport errors in LoggingWebSocketHandlerDecorator .
Removed log.warn ( ) and replaced with log.info ( ) in TitanBlueprintsTransaction.addVertex ( ) method .
Disable preemption when there are no available slots
Modified DataSinkTask.java to handle degree of parallelism greater than one and added directory creation for output .
Added check for managed profile when switching users
Fixed ResourceHelper.java to correctly check the first character of a resource name .
Removed unused import statement for MILLISECONDS in PopulatedCachesTest.java.
Added check for generic type in LinkRenderer to avoid useless warnings
Added dependency requirement to DependencyParseAnnotator
Reverted ternary condition for coverage file path in GoTestRunningState .
Updated IndicesOptions class to clarify wildcard expression handling and allow ignoring of missing indices
Added GBM _ _ to Log.Tag.Sys enum
Update DebuggerManager.java to use PositionManagerFactory extension point instead of deprecated register / unregisterPositionManagerFactory methods .
Update to MaterialViewPagerHelper registration for RecyclerView
Added import for KtCodeFragment and updated TargetPlatformDetector to handle code fragments
Added getDisplayName ( ) and query ( ) methods to ContactSourceService interface .
Added license header to TikaImplTests.java
Added Python language support to CompletionWeigher
Removed unused &apos; headers &apos; field from AppleBundleDescription and AppleBundleBuilder .
Added private constructors to prevent instantiation of utility classes FileUtils and PathUtils .
Removed unnecessary debug statement and simplified code in Source.java
Added logging for failed parsing of conditional values in ConditionalTagsInspector .
Added support for multiple roles in Person model .
Fixed row height setting in AbstractProjectViewPSIPane .
Added Netty channel allocator option to improve performance .
Modified RoboSpiceContentProvider to use ITEM instead of DIRECTORY for content URI patterns .
Added unused declaration suppression for PythonPathCache constructor
Delete @ AutoValue.Validator annotation and replace with @ Deprecated idiom .
Renamed ECC to EC in SSLHostConfigCertificate
Added Google Analytics ScreenViewHit import and updated route for Frames Handler .
Added Trakt API key to ShareUtils and updated ServiceManager initialization
Added Jetty HTTPS support for Admin Console Plugin and disabled sending Jetty version in HTTP headers .
Added import statement for HighlightingLevelManager and added a check to use it in collectIntentionsFromDoNotShowLeveledInspections .
Modified Function.java to pass correct parameter types for varargs methods .
Added mPlayingState = STOPPED ; to set the initial value of the animation
Update episode sorting and formatting
Added FramesHandler.Frames export method with optional Job and path overriding .
Removed testSelectRandomTwoTimes and added testSelectArithmeticOperatorInWhereClause in TransportSQLActionTest .
Added user ID and member count to TagFilterExposer
Added test for FileSystemResource &apos;s URI scheme
Fixed issue with AutoCompleteTextView &apos;s attach / detach handling .
Fix bug in Futures.java : Remove unnecessary ExecutionException with actual cause
Added check for voice capability before returning phone type
Added timeout handling for input dispatching in ActivityManagerService .
Added CrateException handling for unexpected result column symbols in PlannerContextBuilder .
Fixed bug in IronEngineCoolant.java where liquidCoolants were not being returned properly
Update VoldemortConfig to use MySQL database
Removed unused import statement for CRC32 and updated checksum calculation in CompressedSequentialWriter .
Added support for specifying a custom NameResolver for address resolution .
Fixed extractEnergy method in TileEngine to use currentOutput instead of extracted energy for constant power .
Fixed issue with CacheManagingDrawTask.java - removed synchronized block and added check for null before updating cache timer .
Added voice search functionality to SearchDialog by stopping existing search and launching voice search intent .
Update installer GUI to support Eclipse and Netbeans updates
Fixed ZPlan.java closeAllConnections ( ) method to properly handle exceptions and added a finally block to ensure all connections are closed .
Added support for native file system on Windows 2000 and newer .
Set border and viewport border for ConferenceCallPanel
Modified Direction.java to change the implementation of the &apos; opposite ( ) &apos; method .
Added &apos; arguments &apos; field to Api class
Fixed user unregistration by using DAO to retrieve user object instead of relying on passed-in parameter .
Fix NPE in AtmosphereResourceImpl by ignoring Jetty &apos;s toString ( ) and updating request object .
ExplicitInitializationCheck : Changed comparison from &apos; 0 = = &apos; to &apos; 0.0 = = &apos; for floating-point numbers .
Fixed incorrect binding puzzle in IncorectBindingPuzzleFixedTest .
Remove unused variable &apos; sslPorts &apos; in OptionsConnectionPanel.java
Added check for RECEIVE _ BOOT _ COMPLETED permission when setting job persistence .
Added try-with-resources block to properly close LineNumberReader in getMacAddress ( ) method .
Fixed layout parameters for RecyclerViewPagerAdapter
Added list of request interceptors to InterceptingHttpAccessor
Added method to add custom authorities to User object
Added support for DDL exclusive partitions in ALTER TABLE DROP PARTITION
Added support for encapsulating process information and authentication attempt information in SecurityLog .
Removed unnecessary System.out.println statement in HystrixCommand constructor .
Update SharedPreferenceConstants : Remove unnecessary boolean preference and update cache size limit
Added wait for second node to join cluster in DedicatedClusterSnapshotRestoreTests .
Fixed bug in deleting index
Fixed Google OAuth token handling to avoid storing unnecessary tokens .
Modified GrailsLayoutDecoratorMapperTests to use URIPath instead of Page for decorator mapping .
Added check for dl / dd and dl / dt elements to prevent incorrect parsing .
Added functionality to set hint and additional hints for AnswerPlaceholder
Remove unnecessary log statement in SlidingUpPanelLayout .
Updated tutorial page with new H2O World banner and removed unnecessary text .
Added &apos; setResizable ( false ) &apos; to OperationDialog constructor to prevent window resizing .
Removed deprecated methods ensureWritableBytes and isReadable , and replaced them with non-deprecated versions .
Added nullability annotations to Visibilities class .
Added WorkspaceListener to Export class
Added checkConnection ( ) method to OutgoingConnection class and called it in add ( ) and registerSpillingQueue ( ) methods to ensure that the connection is established before adding envelopes to the queue .
Removed UIUtil.drawFramingLines and replaced with DiffUtil.drawDoubleShadowedLine for rendering line markers in DiffLineMarkerRenderer .
Added support for multiple APN types in TelephonyRegistry .
Fixed bug in TachyonConf.java by removing unnecessary null check
Added support for serializing ORecordId as string with &apos; # &apos; prefix .
Removed unnecessary System.out.println statements in AlarmScannerThread .
Added logging statements to KMeansDriver.runClustering ( ) method .
Refresh task list fragment changed to loadTaskListContent ( true ) for better performance .
Added constructor for DefaultNamedDomainObjectSet with ClassGenerator parameter .
Updated JsonPointer and TreeNode classes to use &apos; at &apos; instead of &apos; find &apos; for tree traversal and filtering .
Added Joerg Wuethrich as a co-author to AbstractGCEvent.java.
Fixes # 12345 : Improves offline availability checking for folders
Added check for image.hasMipmaps ( ) before setting NeedGeneratedMipmaps in Texture.java
Fixed issue with HTTPServer not running properly
Added check for registered VCS in VcsDirectoryConfigurationPanel
Added close ( ) method call for FileReader in WiredAccessoryObserver.java
Added support for building Rust binaries with libraries and updated the build process to use the `MakeCleanDirectoryStep` and `SymlinkFilesIntoDirectoryStep` steps .
Added getImageURL method to NetworkImageView to retrieve the URL of the image to be set into the view .
Update security permissions for GUIManager class
Overriding bean definition for bean &apos; : replacing &#91; oldBeanDefinition &#93; with &#91; beanDefinition &#93;
Added DELETE method to UsersResource to delete a user by username , and added PUT method to update user permissions .
Added check for forceSilentReport and endApplication in ErrorReporter # handleException
Fix for GoImportReference to correctly handle PsiDirectoryContainer
Replace material name in ShadowMapNode constructor
Fix issue with invalid service evaluation order
SyncResult : Changed &apos; storage later &apos; to &apos; storage layer &apos; in databaseError description .
Added code to send close frame immediately in FrameHandler .
Remove unnecessary trim ( ) call and simplify empty string check
Added code to dismiss keyguard when showing brightness dialog
Update NodeService to use Version.CURRENT instead of hardcoded version string
Update menu inflation in RecyclerListViewFragment to use menu _ e _ already _ expanded instead of menu _ e _ add _ remove
Added null check for JID node in getRoute method of RoutingTableImpl class .
Added nullability to description field in StreamResponse
Removed unused import and deleted unnecessary tables .
Renamed &apos; add &apos; method to &apos; append &apos; in BasicDBObjectBuilder
Renamed &apos; addQueryParams &apos; to &apos; addQueries &apos; in IonRequestBuilder and RequestBuilder
stats / StatsAbstractInsightsFragment.java : Fix for incorrect date comparison in getDate ( ) method .
Added null check for GenericUDFAbs and updated test case for decimal type .
Added ClassClassPath insertion for Weaver class in Weaver.java
Removed craftbook and sturmeh config options .
Increased number of repositories from 3 to 4 in TestVirtualWebappLoader .
Updated login manager to use user ID instead of account address for authentication failed messages .
EditTicketPage : updated title and description fields to reflect changes in the ticket .
Modified GenericMultipleBarcodeReader to use image.crop ( ) instead of creating a new cropped image .
Removed unnecessary &apos; roots &apos; collection initialization in TreeModelBuilder .
Assume Android NDK is available in AndroidAarIntegrationTest .
Added a check for null edges in the TinkerGraph.currentId incrementation loop to prevent infinite loops .
Added finalizeActions parameter to OutputService.startBuild ( ) method .
Updated documentation for ResourceHttpRequestHandler to clarify usage of ResourceResolvers for front-end needs .
Fixed listen _ address configuration error
Added latch.countDown ( ) in TransportClusterUpdateSettingsAction to signal that the update was successful .
Added test for building mapping with response headers
Fix for # 12345 : Allow non-directory test kit directories to be created .
Replace &apos; topHitsContext.size ( ) &apos; with &apos; topHitsContext.from ( ) &apos; in method buildEmptyAggregation of class TopHitsAggregator .
Removed unnecessary KEY _ SENDER key from Notification.java
Removed unnecessary imports and updated ASM dependencies .
Added import statements for HazelcastParallelClassRunner and QuickTest , and annotated BasicBackPressureServiceTest with @ RunWith and @ Category for parallel testing .
Added extra END _ PADDING byte to FeedableBodyGenerator &apos;s closing state
Fixed bug in partition migration where owner was not being checked properly
Added check for &apos; onScrollChanged &apos; event in onMessage ( ) method .
Added atmosphere.xml configuration file support
Added try-finally block to properly release editor in tearDown method of LocalHistoryActionsTest .
Added a new audio level mapping .
CorruptedFileTest : Changed &apos; testCorrupteOnNetworkLayer &apos; to &apos; testCorruptionOnNetworkLayer &apos; .
Update BTree constructor to calculate maxKeySize and maxChildrenSize based on minKeySize and minChildrenSize
Update JavaUtilLoggingSystem.java to set logger level to FINE .
Removed WIFI _ SLEEP _ POLICY from SETTINGS _ TO _ BACKUP
Removed unnecessary assertion in PolicyBasedAuthenticationManagerTests .
Added support for negative X cubemap face in ShaderCollectionTest .
Added constructor for DefaultExecutionResult
Update copyright year in FileTemplateDefaultExtension.java from 2009 to 2015 .
Removed unnecessary attachment.attach ( attachment ) call in NioEndpoint .
Added stripe factor configuration for MuninnPagedFile
Modified CasLoggerFactoryTests to use canonical path for log file existence check .
Removed unnecessary library loading in NativeMemoryChunk
Added installInput method and removed unnecessary code in Graylog2Module
Added suggestion matches out msg for XMLRuleHandler
Added NdotL variable and updated light calculation for GouraudMaterial .
Added JavaSdkImpl.attachJdkAnnotations ( modificator ) ; to PluginTestCaseBase.getSdk ( String sdkHome ) method .
Simplified toString ( ) method in ListOption class .
Added test case for no matches found in DeepLinkEntryTest .
Modified SendPoiDialogFragment.java : removed unnecessary cast
Added password support for HTTP2 server key store loading .
Added preservation of headers in normal transfer using the body only
Modified ContactListCellRenderer.java to add Dimension import and updated preferred size of nameLabel .
Added support for wrapping action views in MenuItemWrapper .
Removed MediaController initialization in VideoSampleActivity .
Added query runtime check to GlobalOperationHandlers &apos; ReadResourceAssemblyHandler
Added support for additional data types in JSONServiceAction .
Added try-catch block for handling Resources.NotFoundException in getViewId ( ) and getSimpleViewId ( ) methods .
Added origin check for illegal index shard state exception
Added documentation for MethodDelegation and Forwarding implementations .
Added method isEmptyOrSpaces ( CharSequence ) for backward compatibility and updated existing method to call the new one .
Added join clause in SeriesGuideDatabase.java
Replace waiting _ on _ me icon with filter _ inbox icon in CoreFilterExposer .
Update LayoutManager.java to set mRequestPosition to position instead of 222
Increased table size read timeout from 1000 to 5000 in TableSizeReaderTest .
Added support for parsing % alone as the first argument in LispReader .
Fixed issue with auto-scrolling on Android widgets by removing unnecessary postInvalidate ( ) call .
Replace System.out.println with org.robolectric.util.Logger.debug in SQLiteLibraryLoader
Fixed bug : Removed unnecessary test mode check in ApplicationImpl .
Added import statement for &apos; UncheckedIOException &apos; and updated code to use it instead of &apos; IOException &apos; in GradleVersion class .
Added getParentEdge method to PartialPlainStreetEdge class .
Added support for wireless charging in SpiceProfilingUtil .
Added gray and italic coloring to broadcast messages
GitHub Checkout Provider : switch from &apos; master &apos; to &apos; origin &apos; branch for cloning repositories .
Added TODO comment for eliminating spinner and caching in Desktop load
Added animation for item pressed in GuidedActionAdapter .
Allow navigation bar to be shown again after user interaction
Remove close method from ClusterStateObserver
Added secure mode check for user in updatePublicMode ( ) and isSecure ( ) methods .
Fixed bug in MetaDataService where existing mappers were not being updated correctly .
Added logging statement for local node addition to active nodes list
Added import statements for Java Utility classes and changed getSnapshot method to return a List instead of an array .
Added default sound for incoming and outgoing calls
Update deep learning spirals test
Added setAccuracy method to ShadowCriteria
Added check for null hostname in HostnamePort.java
WebViewClassic : Delegated performLongClick ( ) to mWebView .
Added analytics tracking for reader tag previewed with Map &lt; String , String &gt; properties .
Added handling of VoldemortApplicationException in RoutedStore
Added conditional rendering of stream URL in formatted email alerts
Removed unnecessary code for changing spinner arrow color .
Added methods to set tint alpha and color tint for system UI bars .
Update RmiRouteTest to use Bean instead of Pojo
Update SparkInterpreter to import sqlc . _ instead of sqlc.createSchemaRDD
Added methods for getting cache size , max size , and closing the cache
Modified SessionManager.java to correctly handle user sessions and anonymous sessions .
Update WidgetService to check for null intent before updating widget
Added retina support for tip images by replacing hardcoded image paths with ResourceUtil.getResource ( ) and using HTML img tag with width and height attributes for responsive display .
Added method addURL ( URL url ) and updated loadClass ( String name , boolean resolve ) to use super.addURL ( url ) in ModuleClassLoader .
DiscoveryClient : Update instance status when unregistering
Added lastLoggedInFingerprint field to UserManagerService.java
Added support for passing an array of button titles to UIActionSheet constructor .
Removed unnecessary System.out.println statement in GLMModelV3.java.
Added catch block for Exception to handle unknown errors during GELF message handling .
Update ZenDiscoveryTests to use new minimum compatibility version
Removed unnecessary code for sign in switch
Removed unused import statement for &apos; PrintStream &apos; in GrailsUtil.java.
Fix grammar mistake in DropUserStatement.java
Added static final variable LINE _ SEPARATOR for line separator and updated printLineSeparator ( ) method to use it instead of hardcoded value .
Added support for transition effects in JazzyListView .
Update SmoothStreamingManifestParser to return MimeTypes.AUDIO _ E _ AC3 for ec-3 or dec3 fourCCs .
Increased tab max width to Integer.MAX _ VALUE and updated measurement logic .
Removed unused imports : EncodingManager , GraphExtension , GraphHopperStorage , RAMDirectory , GHPlace , GHPoint , GHPoint3D , AASTORE .
Added method seq ( ) to ArrayUtils class and updated Vec.java to use it instead of throwing an exception for non-enum vector identity transformation .
Added testnet seed for Andreas Schildbach
Modified AssetFile.java to use FieldTypeDatabase.getInstance ( ) .getNode ( path.getTypeID ( ) , typeTree.getUnityRevision ( ) , false ) instead of FieldTypeDatabase.getInstance ( ) .getNode ( path.getTypeID ( ) , typeTree.getUnityRevision ( ) ) for better performance .
Suppressing unnecessary warning messages from LWJGL .
Added import for ModalityState in GoModuleLibrariesInitializer
Added &apos; TODO ( gpang ) : Consolidate this array of services with the one in Format.java &apos; comment to AbstractLocalTachyonCluster.java
Implemented impute missing feature for categorical variables .
Added method getStorageLevelAlias ( long ) to class StorageDirId , which returns the StorageLevelAlias corresponding to the given StorageDirId .
Fixed Git freeze issue by updating local changes before refreshing .
Added getAllAddress method to DBTCPConnector and Mongo classes .
Changed settings for data activity timeouts
Added support for Python 3 nonlocal keyword in CompatibilityVisitor .
Removed unused import statement for org.bson.types.ObjectId
Fixed exception handling in RecoverySource.java
Added methods cancelDrag and dragRecipientEntered to BridgeWindowSession class .
Added cast to silence unchecked warning in PyUtil.java
Fix for IndexNotReadyException in FinderRecursivePanel
Added null check for matchWithResult method in ComplexNodePattern class
Fixed uncaught exception handling in TachyonWorker
Added support for NopStmt by introducing a new nop ( ) method .
Throw illegal argument exception for invalid language code
Fixed trimming of username and password fields before login .
Increased size of LruCache from 250 to 1000 and added synchronized block for loadIdToSafeHash to ensure thread safety .
Optimized isMatchingFieldError method in AbstractErrors class by using charAt instead of endsWith for better performance ( SPR-11304 , VESC-165 )
Added check for empty list of launchables before setting task requests
Update DataHeaderFlyweight fields order and offset
Fix for sending broadcast intent to all users .
Update JavaVersionRule to support newer Java versions
Added null check for controller in MediaplayerActivity
Update PythonInPlaceBinary.java to include AddToRuleKey annotations for script , linkTree , components , and python fields
Fixed incorrect printing of variable name in IgnoredJUnitTestInspection .
Update Filegroup.java to use PackageIdentifier.getPathFragment ( ) instead of Label.getPackageFragment ( ) .getRelative ( ) for computing the package-relative path of a filegroup &apos;s attr .
Added library model support in UIRootConfigurationAccessor .
Added logging statement for access denied directories and updated canRead ( ) check for API level 9 and above .
Remove unused variable &apos; mActivity &apos; in Drawer.java
Added OnVideoSizeChangedListener to update video surface view size
Added protection against concurrent modification while checking for ongoing recoveries .
Added try-catch block for removing add-on loaders and added close method call for add-on loaders .
Added check for isUnsubscribed ( ) before requesting ( 1 ) in OperatorMerge and OnSubscribeFromIterable
Set persistent presence away status with a default message if none provided
Added test case for empty tokens block in ToolSyntaxErrorsTest .
Removed unnecessary ContainerResponse check in JerseyBroadcasterUtil
Added logic to mark beacons as inside when they are already inside and their last seen time is in the status preservation file
Added support for XML tags in XHtmlHighlightingLexer .
Remove Singleton annotation from PlanOptimizersFactory
Updated Antlr4Mojo.java to add new parameters for controlling ANTLR &apos;s message format and verbose DFA output .
Fixed auto-increment clause in HsqlDatabase.java
Added test code for selected app mode in SettingsBaseActivity and fixed issue with default application mode not being selected in NavigateAction .
Added begin and end methods to BuckTracing class
Renamed method getScrollableViewScrollPosition ( ) to getScrollableViewScrollPosition ( ) and made it protected .
Added imports for ConfigurationProperties and EnableConfigurationProperties
Roboguice : Changed injection target from &apos; view &apos; to &apos; this &apos; in RoboFragment.onViewCreated ( ) method .
Added method getAsText ( ) and getAsShortText ( ) to AbstractReadableInstantFieldProperty
Added NotThreadSafe annotation to CompilerDaemonManager and made it volatile .
Removed &apos; openProjectInNewWindow &apos; menu item from ProjectPopupMenu .
Update CubeMapMaterial.java to use V.xyz - uCameraPosition instead of normalize ( V.xyz - uCameraPosition.xyz ) for eye direction .
Added method getEntity ( ) to RequestParams class .
Added Priority.HIGH to SuspendContextCommandImpl in DebuggerSession.java
Allow creating hard links for relative source paths
CoyoteOutputStream.java : Changed ob.write ( s ) to super.print ( s ) in print ( String ) method .
Added prepareAlias method call in RootImpl.java to improve performance .
Added ability to send buffers to receiver thread .
Update BooleanType.java and PostgresTypeConverterTest.java to use literal boolean values instead of string literals .
Added support for Action Bar in DemoActivity .
Added check for moved sync data folder and updated UI event for SyncAccountService .
Modified HandleSynchronized.java to change field modifier from final to private and final
Update BufferUtils.java to use Apache Harmony &apos;s DirectBuffer free method instead of Java &apos;s DirectByteBuffer free method .
Added support for finding peers through DNS A records using community-run DNS entry points .
Disable WADL feature and add JacksonJaxbJsonProvider and JsonProcessingExceptionMapper registrations .
Added check for uninitialized collections in CollectionType.renderLoggableString ( ) method .
Added list of RouteBuilderRef to CamelContextFactoryBean
Updated StorageDirView to use HashSet instead of Set for blocksToMoveIn and blocksToMoveOut .
Update distribution repositories to point to services.gradle.org instead of downloads.gradle.org.
Added method isDeveloperVersion and updated getVersionForTracker to check for developer version
MainMenuActivity.java : updated onScrollChanged method to check for null child before returning
Added import statements for ArrayList , Date , List , and Set in JettyRun.java.
Removed getter for Answers enum constant , use direct access instead .
Modified UrlImageViewHelper to call callback.onLoaded ( ) with null imageView when the image is not loaded
Added support for drawing a transparent bitmap on top of a opaque background in View .
Fixed issue with publication configuration in IvyPublication.java and IvyConfigurationContainer.java
Added method &apos; continueOnErrorForInsert &apos; to return whether ( batch ) inserts will continue if an error occurs before the end , and updated &apos; toString &apos; method to call new method .
Removed unused trip pattern update in TestRequest .
Added TableColumnModelListener to stop cell editing when column margin changes .
Added support for Penn Treebank language model in NegraPennTreeReaderFactory .
Added test for onBackPressed ( ) and finish ( ) methods in ShadowActivity .
Added setContentLength method to CacheManager
Minor logging changes in TrackableSession.java : changed logger level from info to debug for tracking-related log statements .
Added &apos; ignoreNulls &apos; field and updated &apos; setProperty &apos; method to respect it .
Added import statement for &apos; FileUtil &apos; in &apos; GroovyScriptRunConfiguration &apos; class .
Renamed &apos; HeaderViewHolder &apos; to &apos; LibsRecyclerViewAdapter.HeaderViewHolder &apos; and made it a static class .
Added import statement for NativeServicesTestFixture and initialized it in ForkingGradleExecuter .
Added verifyWriteAccess method call to ensure proper authorization for setting properties on vertices .
Fix for ANTLRv4 : Skip off token channels reverse correctly .
Added TimeUnit.MILLISECONDS parameter to record ( ) method in DynamicTimerTest .
Disallow Unicode characters in the range 0x0000-0x001F other than 9 , A , and D in XML
Renamed texture wrap mode from &apos; Wrap &apos; to &apos; Repeat &apos; .
Fix edge target retrieval for hierarchical graphs
Added profile storage on ; and removed create user TestUser identified by passwd ROLE &#91; &apos; reader &apos; , &apos; writer &apos; &#93; ;
Modified TwitterRestClient.java to change GET request to POST request in post method .
WebSettings : Remember we checked the Gears permissions and fixed the path to the Gears library .
Added check for isInEditMode ( ) before calling super.onMeasure ( ) in RenderSurfaceView .
Added clearPrecomputes ( ) method call in CellChange.apply ( ) and CellChange.revert ( )
Added serialVersionUID and created constants for true and false values in PactBoolean class .
Modified UrlMappingsHolder to use sets instead of collections for parameter names .
Added SonarQube copyright and license headers to ActiveRuleQuery.java.
Update EnglishUdLas to 88.72648417258083
Update GetOnRootDocIT.java to use JSON streams for better performance and lower memory overhead on the server side .
Update DocumentLoader.java
Fixing bug in Task.initialExecutionResourcesExhausted ( ) by removing unnecessary thread check .
Added constants for special characters in Mariadb generated classes .
Removed unused properties Luminance1 , Luminance2 , Luminance3 , Luminance4 , and Luminance5 from JmeCrossHatchFilter .
Added constructor to MillisType class in DurationType.java
Added import statement for AnActionEvent and updated forceSyncUpdate method in ShowBaseRevisionAction .
Added method dohasheq ( Object o ) for hashing objects that implement IHashEq .
Added fallback package ID for voice recorder activity .
Fixed logging message in UserDataProperty.java
Added check for existence of dataset before downloading it
Added import statement for StatefulEJBLocator and updated code to use it when creating a SessionID .
StrictDuplicateCodeCheck : simplified code duplication check
Added license headers to Utils.java
Fixed issue with credential ID being blank
Removed unused import statement for ZWaveEndpoint
Disconnect from MySQL database using connection.close ( ) instead of isConnected ( )
Ignored exception in Clock2.java
Added timeout to ManagedShutdownStrategyTest
Added testQuantifiedPredicates method to PredicateTests class and added a new test case for quantified predicates in jOOQAbstractTest
Removed unnecessary variable declaration in LayoutTypePortletImpl .
Updated jump to commit tooltip to include commit hash and root name for multiple roots .
Renamed import from &apos; com.intellij.psi.templateLanguages.TemplateDataElementType &apos; to &apos; com.intellij.psi.templateLanguages.ITemplateDataElementType &apos; and added static import for &apos; IXmlLeafElementType&apos;
Added collection initialization for races field in SQLSelectTest .
Added documentation and notes for MultiUserChat room ( XEP-45 ) and its affiliations and roles .
Added conversion of instant to chronology in AbstractPartialInstant constructor .
Added logic to calculate lanes for going straight , and updated TurnType.java to include new slight turn types .
Removed unused import android.view.inputmethod.InputMethodSubtypeArray ; and changed additionalSubtypes to additionalSubtypesMap in InputMethodInfo constructor .
Update ContextStorageOverride class name to io.grpc.override.ContextStorageOverride
Set IME button visible for current input method in status bar .
Allow non-read-only direct buffers to be wrapped .
Remove unused field &apos; visibilityParseContext &apos; and method &apos; parseVisibilityTarget &apos; . Simplify constructor and remove unnecessary parameters .
Minor change in logging level for syslog message parsing errors .
Remove unused configuration key
Modified PythonTestRunConfigurationsConverter to use PythonTestConfigurationType.getInstance ( ) .getId ( ) instead of hardcoded ID
Removed unnecessary comments and implemented the WrapperListAdapter interface to wrap the asymmetric grid adapter .
Added methods for setting key bindings : setBindings ( KeySequence , BindableCommand ) and setBindings ( List &lt; KeySequence &gt; , BindableCommand )
Added notifyActionStartedForState and notifyActionFinished methods to SlidingLayer class .
Fixed issue with bundle context injection in WebContextLifecycleInterceptor .
Fixed issue with chart panel not being properly invalidated when maximized
Fixed ParallelAction to correctly check for actor existence before calling act ( ) on child actions .
Renamed JavaScriptAggregatorFactory and CardinalityAggregatorFactory to use input instead of name .
Added efficient String retrieval method getStringOrNull to ResourceBundleMessageSource .
Fix timestamp comparison in RowSeq.java to handle cases where new timestamp is less than previous timestamp
Added onApplyThemeResource method to apply theme resources in PreferenceActivity .
Added ZWaveConfigurationCommandClass to CONFIGURATION command class .
Added space to ban list message
Fixed cache name in UserImpl.java from &apos; userame2roster &apos; to &apos; username2roster &apos; .
Fixed issue with focus transferring in IdeEventQueue .
Added fairness check for DiffIterableUtil.fair ( ) method
Added OnGenericMotionListener support to ReactRootView
Update AbstractBlockChain.java to use SPVBlockStore in lightweight SPV mode instead of BoundedOverheadBlockStore .
Modified TaskStack.java to account for docked stack &apos;s effect on other stacks &apos; size
Convert JavaExtensions class to interface and remove unnecessary private constructor .
Modified TieredBlockStore to use belongTo instead of equals for location comparison .
Replace add with replace in StatsAbsListViewFragment
Removed unused import statement for IOException in ProxyHTTPSPDYConnection.java
Added &apos; WritableDebugStatus &apos; interface and implemented it in &apos; K9MailLib &apos; class .
Fix grammar mistake in ESIntegTestCase.java
Added UnderFileSystemCluster field to AbstractLocalTachyonCluster
Throw a TwitterException with a detailed message when Twitter returns a non-XML response .
Added &apos; ClobOpenCount &apos; to DruidDataSource.getMetrics ( ) method .
Added null check for key in SymmetricEncryptionConfig.getKey ( ) and updated setKey ( ) method to handle null keys .
Fixed issue with ArrayRenderer &apos;s remaining elements calculation .
Added assumption for NIO2 in FileAttributesNio2ReadingTest .
Added two new exclusions for MediaScannerConnection and DevicePolicyManagerSettingsObserver
Remove unnecessary discrimination check for deactivated items in DownloadIndexesThread .
Added fallback empty string return for iOS charset decoder when input byte sequence can &apos;t be encoded .
Fixed issue with cell reset command in ResultSetViewer
CacheAspectSupport : Added a suggestion to use a specific CacheManager as the default one when multiple CacheManagers are defined .
Added functionality to add calendar resources
Fixed syntax highlighting for Python f-strings
Added logging and error handling for reINVITEs while the dialog is not confirmed
Removed unnecessary variable &apos; pos &apos; in DashRule.java
Toggle debug logging and signal strength tracking preferences
Added support for analyzing expression types in WHEN clauses
Allow bootstrap resources in BootstrapClassLoaderUtil .
Added PythonDocstringTest to PythonAllTestsSuite
Modified ColorPicker.java to adjust color wheel radius and translation offset .
Added use of System.nanoTime for relative time measurement in Cache class .
Remove unnecessary message retrieval in MultiUserChatTest
Change variable &apos; size &apos; from int to long in CombineHiveInputFormat.java
Added userId parameter to KeyguardViewMediator.onUserSwitchComplete ( ) and added logic to dismiss keyguard when switching to a guest user .
Removed unused import org.apache.ivy.util.Message ; in IvyXmlModuleDescriptorParser.java
Removed unused variable &apos; imgBlavatar &apos; in ReaderPostDetailFragment .
Added copyright and license headers to CertificateValidator.java.
Added sorting and flattening of parameters for ZXingTestActivity .
Added @ ExcludeIn ( CUBRID ) annotation to Order _ NullsFirst and Order _ NullsLast tests in SelectBase.java
Added TitanGraph and TitanProperty to LoaderScriptWrapper &apos;s whitelist
Added methods for setting and getting player list name , and updated setDisplayName method to check for conflicts with player list names .
Added @ SuppressWarnings ( &apos; unchecked &apos; ) annotation to SpringExpressionsParser.java to avoid unnecessary warnings .
Removed DuplicateStringLiteralInspection from the list of registered inspections
Added support for displaying available columns in the data table
Disable user agent string in WebViewActivity and NotificationsWebViewActivity to avoid browser-sniffing issues .
Added constructor for StringList that takes an array of objects and initializes the list with their string representations .
Removed print statement for shader sources in TestShaderNodes.java
Added vehicle and key parameters to GraphHopperWeb URL
Set endpoints for shared actor
Added timing for graph building
Reset calendar to either today or the specified date
Added import statement for com.google.common.collect.ImmutableSet ; and replaced Collections.unmodifiableSet with ImmutableSet.builder ( ) .addAll ( seedProvider.getSeeds ( ) ) .build ( ) in getSeeds method .
Added private constructor to prevent instances of StringAndPartitionAwarePartitioningStrategy
Removed unused import : ManufacturerUtil
Removed unused import statement for &apos; IntrospectionUtils &apos; in JIoEndpoint.java.
Modified BaseOutputWriter.java to replace all spaces , double quotes , and single quotes with underscores .
Added a comment to note that saveParserToTextFile ( ) method should use character encoding .
Fixed issue with duplicate onNext calls in OperatorMergeDelayErrorTest .
Renamed connection ID from &apos; conn &apos; to &apos; conn * &apos; to avoid conflicts with MongoDB server-assigned IDs .
Added import statements for VirtualFile and related classes .
Added @ Ignore annotation to faultTest method in MasterFaultToleranceIntegrationTest.java to resolve issue with HDFS as UnderFS .
IoUtils.java : Quietly closes the given Closeable closeables . Any exceptions thrown by close ( ) are swallowed .
Update ExecutionTimeLimitTest to use new server URL
Update RatpackLauncher to use configure ( ) instead of config ( )
Added TestLogging annotation to InternalEngineMergeIT test class .
Added test for not found returns empty
Added null check for packet.getFrom ( ) in FromContainsFilter.accept ( Packet packet ) method .
Update InsertUpdateTests.java to use assertSame instead of assertEquals for values returned by UPDATE ... RETURNING clauses .
Updated JavaSmartCompletionContributor to use CompletionUtil.DUMMY _ IDENTIFIER _ TRIMMED for dummy identifier
Added line editText.setInputType ( 0 ) ; to set input type for EditText widget .
Added import statement for java.util.Collections and modified OrientElementIterable.java to use Collections.emptyIterator ( ) when iterable is null .
Fix issue with nonexistent parent call ID in ParcelableCall.checkCallTree
Remove unnecessary import statement for KeyStore
Removed unnecessary import statement for LinkedList .
Removed unused method length ( List &lt; Node &lt; Token &gt; &gt; tokens ) and updated Parser.java
Removed unnecessary import of ActionRequestImpl and updated package name for mail util class .
Added handling of primary shard not being allocated or started in SnapshotsService .
Update Model.java to adapt original model to different dataset .
Modified ConfigurationClassParser to use &apos; i &apos; instead of &apos; 0&apos; when resolving placeholders in @ PropertySource locations .
Added support for nested collections in DB.getCollectionFromString ( ) method .
Fix InstanceRegistry # getApplicationDeltas to explicitly specify if a remote region is needed .
Added import statement for CountDownTimer
Modified OQueryOperatorContains.java to use Object instead of ORecordSchemaAware &lt; ? &gt; for collection type .
Update OverridingClassLoader to use org.aspectj and net.sf.cglib instead of org.springframework.core for instrumentation
Fix for handling DSL errors in Groovy scripts
Added setupActionBar and setupViews methods to AddActivity .
Removed unused field &apos; mTachyonHome &apos; from Client class .
Fixing null checks for Query.java
Change interner from strong to weak for PackageIdentifier
Added check for FML installation on connect , and added logging for vanilla kicking player
Added grid definition and hyperparameters to GridSchemaV99.java and updated Schema.java to include ModelParametersSchema.class
Added support for Bluetooth downloads in DownloadManager .
Update copyright year in FreeMarkerConfigurationFactory.java from 2008 to 2009 .
Added clearCapabilities method and transport requirement to NetworkRequest
Added retry logic for MetaStore DB connections and improved error handling .
Removed Sun / Oracle-specific APIs in VirtualMachineMetrics.java
Added reload resources for WAR files
Fixed typo in TextView.java : changed &apos; setText ( getText ( ) , ... &apos; to &apos; setText ( mText , ... &apos; .
Added error handling for decoding subjectAltName in TLSStreamHandler
Added method to Node class for getting string representation and fixed Parser to handle &lt; html &gt; tags correctly .
Fixed Utils.java to handle ParseException and return -1 instead of crashing
Remove unnecessary @ SuppressWarnings annotation in StaticListableBeanFactory and SimpleJndiBeanFactory .
Updated WallpaperService.java to set mReqWidth and mReqHeight in onDesiredSizeChanged ( ) and added doOffsetsChanged ( ) call .
Removed unnecessary stateUpdated call in RuleEngine.stateChanged
Add context specific weld container reference for Jetty deployment
Remove Debug listener in case of disconnect by QUIT or ERROR
Added support for multiple method separators in HTMLTextPainter .
Added support for suspendable var setters in SuspendableHelper .
Added counter variable to keep track of unified tokens in Unifier.java
Added support for punctuation in NaturalLogicAnnotator .
Removed static modifier from createTemplateFile method in RawLogBuffersFactory .
Allow touch events for live wallpaper on Android 7 and later .
Renamed &apos; Change details &apos; action to &apos; Change Details &apos; in ChangesViewManager.java
Replace manual sequence number assignment with Entry.getSequence ( ) in BatchEntryConsumer .
Added dependency on rstudioapi package in DependencyManager .
Added MySQL driver loading and updated Guice listener to use JDBI .
Fix for unshare operation when file is deleted before
Added import statements for File , FileInputStream , FileNotFoundException , IOException , and InputStream in CamelSalesforceMojoIntegrationTest.java.
Fixed bug in FileMetadata.java : removed unnecessary &apos; afterCR &apos; variable and simplified logic for handling EOL characters .
Added support for displaying TV contents with TvView . Now applications can show TV programs from various TV sources that implement TvInputService . Also added a way to dispatch input events to the connected TvInputService in order to enable custom key actions for the TV input .
Added wait for ES to be ready before starting SonarQube server
Added scheduled ping to NettyTransport
Added closing flag for collapsing panel
Added log statement and try-catch block to RunTests.main ( ) method to handle Grails bootstrapping and session management properly .
Simplified proxy settings in GithubConnection
Added dependency tracking for external extensions
Added method getUfsPath ( ) to TachyonFile class
Added Singleton annotation to Navigator class .
Added email validation to EditTextPreferenceWithValidation
Removed unnecessary imports and added missing import for Background annotation .
Added CachingPatternSpecFactory and prevented it from being added as a dependency to the Tooling API jar .
Deprecate unused &apos; connector &apos; field in Response class .
Updated label text for password and port fields in FirstWizardPage .
Added functionality to suppress misspelled suggestions in Match class
Added logging statement in PrivacyService to check replaced restriction and method
Added support for Leopard-style search field in SearchTextField .
Fixed bug in OperatorOnBackpressureBuffer.java where capacity was not being incremented correctly .
Added import statement for ArrayUtilRt and modified dropTarget validation to use it .
Removed unused method captureViewInfo and related code .
Remove test provider { provider } from LocationManagerService
Optimized rendering configuration by adjusting frame update rate and cordon time .
Removed unnecessary synchronization in cleanupOnChannelDown method .
Added JavaDocLocalInspection with ignoreDuplicatedThrows = false to JavadocHighlightingTest
Added override of toString ( ) method in SystemLogRecord class .
Added code to save cookies into local store and remove expired cookies .
Modified PlanPrinter to correctly handle original constraint and domain summary for TableScan .
Added didRequestLayout and related methods to ShadowView
Fix for resolving qualified names with prefixes
Update Frame.java to use decode ( ) method instead of decode ( ChannelHandlerContext , Channel , ChannelBuffer ) and remove unused extractFrame ( ) method .
Modified SourceFormatter.java to update regular expressions for jsp include files and language keys , and added support for portal source formatting .
Added cluster.start ( ) call in NotSoMiniClusterIterations and StreamingScalabilityAndLatency test classes .
Added laser data and visibility flag to TileLaser .
Fixed division by zero issue in MultiUserChatServerImpl.java by removing unnecessary calculations .
Added null check for instance in setKeyStorePassword method .
Added check for data set before calling listener methods .
Added stack dump for debugging purposes in ShadowActivity # findViewById
Added support for remote video providers in RemoteConnectionService
Updated MongoOptions.java to use Socket # setSoTimeout ( int ) instead of Socket.setSoTimeout
Modified TaskManager.java to update the description of a task that is already running and added a new task to the list of tasks to start .
Added ClosedChannelException handling to AioSocketChannel .
Fixed issue with incorrect region name extraction in IndexUploader.java
Modified CubeImageView to handle null drawables and updated image loading logic .
Added cancelAllSuperActivityToasts ( ) in ProcessHyperlinkAction and SelectionBookmarkAction to cancel any existing toasts before showing a new one .
Throwing a RuntimeException instead of calling System.exit ( ) when there are issues with parsing properties .
AnimatedVectorDrawable : fix for pause and resume issue .
Replace SOFT _ CONCURRENT with CONCURRENT in PageCacheRecycler.parse method .
Fixed issue with Postgres refcursor types by adding separate queries for autocommit transactions .
Added support for monitoring presence using ISON , WATCH or MONITOR and added away user mode .
Fixed issue with third-party reconnect / reassociate while temporarily disconnected for p2p
Added fetchPlan field to JSON serialization of ORecordAbstract and updated JSONTest to use new format .
Subsumed testFirstAndLastLeg method in TestRequest class .
Removed unused method readHistoricalDataImpl ( ) in ActivityChooserModel.java
Added texture support for PShape3D .
Fixed issue with page size being too small for image &apos; + name + &apos; . Now throwing a GdxRuntimeException if the image is larger than the page size .
Added test for unary call response size
Added trace logging to StandardIDPool # getID
Added Serializable interface implementation to DefaultIdeaContentRoot class .
Update ClusterMemoryManager to use reserved memory pool for biggest query
Removed unused import &apos; com.google.gwt.json.client.JSONArray &apos; in FindResult.java
Update import statement for Arrays from edu.emory.mathcs.backport.java.util.Arrays to java.util.Arrays
Ensure that the current page is within bounds
Added onReset method to stop battery receiver .
Removed unnecessary code in DatabaseEditorInputFactory.java
Added animation to dismiss button in SweetAlertDialog
Removed unnecessary check for rectangle crop shape in CropOverlayView .
Update PsiTestUtil.java to use org.jetbrains.annotations.NotNull and add missing imports
Fixed assertion in AbstractJPATest # testQueryWithKList
Added method getCurrentRecipe ( ) to TileAutoWorkbench.java
Set active to true when marking last write
Makes host : localPort server available on remote side as localhost : remotePort and vice versa .
Added sendToTarget method to ShadowMessage class and added test case for it in MessageTest class .
Update user agent string in ApacheAsyncHttpProvider and JDKAsyncHttpProvider
Updated logging statements in HttpUtil and XMPPConnect classes .
Fixed RedissonLockEntry constructor to use Semaphore ( 0 ) instead of Semaphore ( 1 )
Rename BinaryBlockWriter to writer in DeprecatedLzoBinaryBlockRecordWriter
Changed time range type from relative to keyword
Added check for empty or null target hostname in authenticateDomain method .
Added support for xz compression codec in CommonCrawlNGramJob .
Removed unnecessary import statement for &apos; PrivateApi &apos; in PacketTransceiver.java.
Refactor AbstractTFS.java to address HBase support issue ( TACHYON-27 ) and add TODO for new internal API support ( TACHYON-46 )
Modified PhoneWindowManager.java to change the behavior of passing keys to the application when the device is in a non-interactive state .
Added option to specify weakly linked frameworks and updated documentation for -frameworkpaths and -resources options .
Added idempotent key to FtpConsumerIdempotentKeyChangedIssueTest
Update Kryo serialization registrations
Added reset ( ) method to OStorageLocal to fix wrong record type issue .
Modified StreamError.java to use halfOpenElement instead of rightAngleBracket for xmlns attribute .
Added ability to add component items to disco # items list and updated entity discovery for better performance .
Added method getAuthenticationDetailsSource to AbstractProcessingFilter to address SEC-310 .
Added import of processing.opengl package in AndroidPreprocessor .
Renamed UDFCONTEXT _ SCHEMA _ KEY from &apos; schema &apos; to &apos; cassandra.schema &apos; in CassandraStorage.java
Fixed issue with duplicate variable insertion in ActivitiOptimisticLockingException test .
Added casts for integer fields in OClass and OSchema to fix issues with deserialization .
Added &apos; spiegelte &apos; and &apos; spiegelten &apos; to WiederVsWiderRule .
Added MoveMessage class to StorageLoadBalancer
Added null check and improved logic in GzipUtils.java
Fix for temporary ImmutableList implementation and endianess in StreamProcessor .
Updated ManifestUpdateMojo to use &apos; android.manifest.versionCodeAutoIncrement &apos; instead of &apos; android.manifest.versionCodeAutoincrement &apos; for the parameter expression .
Added base64 encoding for success data in SASLAuthentication .
TargetElementUtil.java : Removed unnecessary check for PsiReference in findParentElement method .
Added a backup stream for lazy list to prevent data loss during modifications .
Added support for real-time segments in FilterPlanNode
Added canDisconnectAndKeepPlaying method to MediaRouteDescriptor to indicate whether the route can be disconnected without stopping playback .
Added &apos; cloneForSpatial &apos; method to PhysicsHoverControl and TestPssmShadow classes .
Renamed field &apos; field _ 804 _ d &apos; to &apos; hoverStart &apos; in CoreProxy.java
InstallDirect : Remove unnecessary environment variable setting and add export command for NO _ UIPRINT .
Deprecate &apos; addConstructorArg &apos; in favor of &apos; addConstructorArgValue &apos; and addindexed constructor arg value .
Increased entryPointSize from 20 to 100 in OTreeMapPersistent
Added comma after world in Location.toString ( ) method .
Removed unused import statements in RestIndicesExistsAction.java
Added check for static keyword in PsiModifierListImpl
Added support for rate ( ) function with optional counter and countermax parameters in GraphHandler.java.
Added test texts for Stepic connector
Modified StandardWrapperValve.java to correctly handle comet requests .
Added random seed for new wallets
Improved EarClippingTriangulator : added support for convex polygons and improved performance . Also , fixed several bugs and updated documentation .
Added import statements for com.google.bitcoin.core.Utils , com.google.protobuf.ByteString , and com.google.protobuf.MessageLite , and updated the timer mechanism to use a shared timer for expiring connections .
Added import for org.junit.Assume ; and updated TestMetrics test to use Assume.assumeFalse ( Os.isFamily ( Os.FAMILY _ WINDOWS ) ) to skip metric usage tests on Windows .
FieldSortBuilder : Renamed &apos; ignoreUnampped &apos; field to &apos; ignoreUnmapped &apos; , and fixed typo in &apos; ignore _ unmapped &apos; field name .
Added test cases for contains with inexistence and null values .
Fixed incorrect fq name validation by allowing &apos; / &apos; and &apos; * &apos; characters .
Reloading local messages for a given folder
Renamed class name from &apos; ShellException &apos; to &apos; ShellException &apos; and updated the message format to use &apos; getSimpleName ( ) &apos; instead of &apos; getName ( ) &apos; .
Added support for YUV _ 420 _ 888 format in Type class .
Renamed inner class UltimateRecyclerviewViewHolder to UltimateViewAdapter
Added support for Enterprise mode in NeoServer
Replace Iterators with Iterators from guava
Added null check for cause parameter in setFailure0 method .
Updated DateUtils to use a more lenient date parsing strategy and fixed timezone issues .
Update LanguageToolSupport to use new language model rules and fix shutdown issue .
Added getCullingArea method to Group class .
Added import statement for com.intellij.openapi.util.io.FileUtil and updated withWorkDirectory method to set PWD environment variable on Unix systems .
Added pause ( ) method to ShadowFragment , which calls onPause ( ) on the real fragment and sets resumed to false .
Added anonymous user support to AuthToken class .
Removed unnecessary logging statement in SpiceManager.java
Added &apos; ancora &apos; option to SpanishTokenizer .
Removed unused import statement for &apos; org.apache.camel.Endpoint &apos; in RabbitMQConsumer.java.
Fixed testToArrayCursorClosed to use snapshot with hint instead of batchSize ( 3 )
AddLocalExchanges : Changed planAndEnforce method to use singleStream ( ) instead of defaultParallelism ( session ) for filtering source .
Added public modifier to convertToHex method in SecureHashUtil class .
Added support for restoring legacy pipe gates .
Added ConfigManagerConfig and updated MasterMain to use it
Added generateFont ( int ) method to FreeTypeFontGenerator .
Change exception type from Exception to Throwable in Driver.java and Utilities.java
Added a check to ensure that the media player is not null before pausing it in the PlaybackService .
Added static final instance for NoOpProjector
Added test for hard link on Windows
Added nullability annotation to HgErrorHandler # ensureSuccess method
Added factory initialization for top-level aggregators in PercolatorService .
Added method call to flush ( ) in ResponseBodyEmitterReturnValueHandler
Added print statements for debugging in MotionTrack.java
Added null check for mCallback in BaseGetNewsTask # onPostExecute
Added method changeTimeZone ( TimeZone from , TimeZone to ) to JDateTime class and updated TimeZoneTest class to include test case for changing time zones .
Removed unused variable &apos; ticks &apos; and updated code to use &apos; ConcurrentLinkedQueue &apos; instead of &apos; ArrayList &apos; for storing ticks .
Updated CsrfPreventionFilter to trim entry points
Increased length field in PeerAddress from 4 bytes to 8 bytes
Fixed type mismatch in SeekableViewsForTest.java by casting current to double
Added a new test case for PropertyAnnotationsTest to prove that the method we determine the value does not use the `toString ( ) ` method of the enum .
Fixing bug in TachyonFS.java where folder files were not being fetched correctly .
Renamed PyCharm prefix from &apos; PyCharm &apos; to &apos; PyCharmCore &apos; in PlatformUtilsCore.java.
Added logging of IOException messages in ReplicaSetStatus
Added field &apos; mTriggeredExpand &apos; to track whether QSPanel was expanded programmatically .
Removed unnecessary parameter &apos; packet &apos; from method &apos; removeItem &apos; in class &apos; IQRosterHandler &apos; .
Added @ Ignore annotation to Examples test class to skip running it during documentation validation .
Ensure active mobile policy is defined for subscriber ID
Muted NetworkDispatcher exception for connection reset by peer
Added import for StringUtil in BasicDomElementComponent.java
Removed unused constant MASTER _ FORMAT _ FILE _ PREFIX and updated constants
Fixed invalid return type in testInvalidFinders
Added support for marking cookies as HttpOnly in CookieRetrievingCookieGenerator .
Added import statement for IdeBorderFactory
Added documentation for using ExpectedException rule with Test annotation .
UnusedReturnValue : Delete or replace unused return statements
Removed unused variable &apos; concurrentExecutionThunk &apos; and updated import statements .
Added logging for texture atlas size warnings and fixed addGeometry method to return false if atlas is full
Removed unnecessary comments and optimized code for faster execution .
Added support for custom cursors in the gwt-backend
Added support for accessibility state changed notification in AdapterView .
Allow IME to show when selecting non-editable text
Added logging statements to TransactionOutput.java to track when an output is marked as spent or unspent .
Added import statements for FileToArchiveEntrySetTransformer and ZipEntryToArchiveEntryTransformer , and added a new constructor that takes a Transformer &lt; Set &lt; ArchiveEntry &gt; , File &gt; as a parameter . Also , added a new instance method that takes a File argument and returns a Set &lt; ArchiveEntry &gt; . Finally , updated the existing constructor to call the new constructor with a new FileToArchiveEntrySetTransformer ( ZipEntryToArchiveEntryTransformer ( ) )
Added check for empty base name before adding it to basenames list in PluginAwareResourceBundleMessageSource .
Fixed response status in GLMProgressPage2.java
Added method isScreenBrightnessBoosted ( ) to BridgePowerManager.java
Update EndpointWebMvcAutoConfigurationTests to reflect changes in MvcEndpoint bean count .
Removed unnecessary import of LivingEntity in EntityDamagedByEntityEvent and EntityDamagedEvent , and updated package name in EntityEvent .
Added support for tessellation shaders and patches in GLRenderer .
Added method to set color and region for FastImage actor .
Added field &apos; mShareWithUser &apos; to represent share file with user and updated setter method accordingly .
Renamed &apos; AsyncShardOperationAction &apos; constructor to &apos; TransportShardReplicationOperationAction &apos; .
Added null check for property values in EdgeSerializer
Increased boost of &apos; body &apos; field in simple query string from 10.0f to 1000.0f.
Added null check for feed content and threw IOException if content is empty .
Added import for ExternalSystemException and added external-system-rt.jar to classpath
Replace LoggerFactory.getLogger ( this.getClass ( ) ) with LoggerFactory.getLogger ( TicketGrantingTicketImpl.class )
Fixed bug in Moco.java where charset was not being checked for null
Removed unnecessary import of AuthException in AuthConfig and AuthConfigFactory .
Changed color disabled from white to darker gray in FloatingActionButton.java.
Removed unnecessary &apos; System.out.println &apos; statement in PersistenceStressTest .
Skip class references in extends / implements lists during resolution
EntityEditorPropertyTester : Use editor.isDirty ( ) instead of commandContext.isDirty ( )
Replace Charset.defaultCharset ( ) with UTF _ 8 in ResourceShrinker.java
VertxOptions : Disable metrics by default
Added isToString method to Invocation class and updated InvocationTest to use it .
Catching Exception instead of SnapshotException in HeapAnalyzer
Added check for PropertyType.getPayloadSize ( ) - propSize &gt; = newBlockSizeInBytes in WriteTransaction.java and updated TestPropertyBlocks.java test case .
Renamed test root directory from &apos; myTestRoot &apos; to &apos; myTestRoot / testRoot &apos; in GitPlatformTest.java
Fix HikariConfig.java : Minimum idle connection limit cannot be negative or greater than maximum pool size .
Added system API for disabling wallpaper touch events and added method to always read close on touch attr .
Added training sample and epoch information to DeepLearningModel HTML report .
Disable interface and reset data structures in WifiStateMachine .
Increased maximum deviation for secondary read distribution from 3.0 % to 5.0 % in SecondaryReadTest .
Added method getPlaceById to ReactiveLocationProvider , which returns an observable that fetches a place from the Places API using the place ID .
Remove unnecessary / LD flag for position-independent code .
Removed call to requestFBOLayer ( ) in PGL.java
Added constructors and getters for WorkflowId class .
Modified ClosureRewriteClass.java to use hotSwapScript instead of NodeTraversal.traverse
Added getSuccessDescription method to PyPackageManagerUI class
Fixed issue with comboBox table cell editor on Mac . Now , myTable.setValueAt ( myValue , myRow , myColumn ) is called after myValue is set .
Disable @ export has no effect here
Changed condition for running CacheUpdateRunner in single thread mode .
Added import statement for android.text.TextUtils ; and fixed null pointer exception in TvContract.encode ( ) and TvContract.decode ( )
Added resource entry name fallback for unknown IDs in ScalpelFrameLayout
Added BrowserCompatHostnameVerifier to Ion &apos;s HTTP client and updated ExifTests to use raw.githubusercontent.com instead of raw.github.com
Added methods onSaveInstanceState and onConfigurationChanged to Controller and ControllerActivity classes .
Update GLMBasicTestBinomial.java to change residual deviance tolerance from 1 to 2 .
Added fluid rendering support to GuiAdvancedInterface .
Moved mod list from bottom to top of screen in GuiMissingMods
Added code to handle themes in PluginsEnvironmentBuilder .
Modified GrailsResourceUtils.java to return a FileSystemResource or UrlResource with a trailing slash for better compatibility with Grails 3.x applications .
Simplified ReplicatedTransaction.toString ( ) implementation .
WebView : Implemented support for multi-touch events .
Added getClientId method to ProjectUser class .
Update copyright year in FragmentSide.java and change comparison method to Comparing.equal.
Added getCurrentTime method with user-set offsets and updated formatToLocalReleaseDay method .
Added a short sleep statement to give a rest to CPU in MapStoreTest .
Hide keyboard when showing message
Fixed assertion in GravesBidirectionalLSTMTest to use assertEquals instead of assertArrayEquals
Added handling for unbound targets in GraphVisualizer .
Added support for Linux in JavaSdkImpl .
Added method getCurrentTileIndex ( ) and setCurrentTileIndex ( int ) to TiledSprite class .
Change executor service from newCachedThreadPool to newSingleThreadExecutor
Added import statement for ElevatorEdge and updated traverse method to include ElevatorEdge .
Added PsiDocumentManager.getInstance ( getProject ( ) ) .commitAllDocuments ( ) ; in HeavyIdeaTestFixtureImpl.java
Modified ActivityStack.java to change the visibility logic for activities launching behind other activities .
Fixed typo in Intent.java
Added note about using advice from a Java agent with Byte Buddy and registering class file locators for assembly of the advice class &apos;s description at runtime .
Bumped version number from 1.3.0 to 1.3.1 .
Update Netty codec MessageAggregator.java file .
Disable caret and hide gutter for LanguageConsoleBuilder
Added support for 0-shard spec in DruidMasterRuleRunnerTest .
Added sleep statement to allow consumer to delete after done , and added assertions for entities in database .
Added support for selectable preferences
Update license header to Apache License Version 2.0
Update copyright year in FileBasedDeclarationProviderFactory.java
Added import statement for Java Util HashMap and updated getPersistedConfig method to use a mutable map . Also , added a new exception class for widget type not found .
Modified VersionValue.java to return nullable boolean for isUnsaved ( ) method .
Added cookies to Request class .
Fixed issue with phone number retrieval by using getPhoneNumber ( true ) instead of getLine1Number ( ) and also removed unnecessary code .
Added conversions and filter-monadic operations to Value.java.
Update from ActionBarActivity to AppCompatActivity
Fixed registerAction method in KeyParser class to pass correct number of shortcuts .
Added test for getting DataFormat from CxfEndpontProperties
Remove unused import statement for Profiler
Renamed local context name from field name to declaring class name + &apos; / &apos; + field name
SubmitReportWsAction : set report data parameter as required , and add test cases for ComponentDto.toString ( ) and isRootProject ( )
Added public modifier to RenderScript constructor
Added onScrollChanged method to CordovaWebView to log scroll changes
Simplified &apos; GetStatusOptions &apos; class by removing unnecessary code .
Removed unnecessary constructor in RegexpHeaderCheckTest .
Added BuckVersion to ProjectGenerator &apos;s header symlink tree hash code calculation .
Added WARN log level for quorum just reached message in SingletonLogger .
Added method openRawResource ( int , TypedValue ) to BridgeResources class .
Change Utils.hash ( ) method to return a long value instead of an int value
Update CompoundButton.java to apply padding for API level 17 and above .
Added USE _ SERVLET _ CONTEXT _ PARAMETERS to ApplicationConfig
Added test for creating block ID with maximum sequence number .
Modified VoiceRouter.java to change distance threshold for voice guidance
Added import statement for ResolveResult and updated getTargetElement method to handle it .
Removed unnecessary import statement for &apos; com.hazelcast.test.annotation.Repeat &apos; in JoinStressTest.java.
Added null check for foreign key columns in ForeignKey.java
Update ActionButton.java to fix issue with popup menu
Added import statements for JakartaRegexpCapabilities , JavaUtilRegexCapabilities , and RegexQuery classes . Modified test method to include additional assertions .
Set highest object ID to position instead of object ID
Changed recordCursor.acquire ( ) from CHECK to NORMAL and removed unnecessary &apos; inUse &apos; variable .
Fix for bug # 12345 : Changed comparison of JIDs to use &apos; toBareJid ( ) &apos; method for correctness and stability .
Removed @ webref and @ brief annotations from XML.getLocalName ( ) method .
Added member alive method to AbstractReplicatedMap
Added &apos; true &apos; argument to super.setId ( id ) in DeltaSession.java to enable delta request tracking .
Added method chaining for position and truncate methods in ChannelFunctionFileChannel .
Added import statement for ObjectNode and modified executeTaskOperation method to return an ObjectNode instead of void
Added check for unit test mode before preloading actions
Update VirtualFileVisitorTest to use VirtualFileVisitor &lt; Object &gt; instead of VirtualFileVisitor
Added support for decompilers and non-local files in DiffContentFactoryImpl
Added DexInfoTranslator class for translating classes.dex entry
Added method isCancelled ( ) to SafeAsyncTask and modified cancel ( ) method to check if future is null before calling future.cancel ( mayInterruptIfRunning ) .
Update PdfJsWindow.java to override setInitialView and fire PDF load event .
Added import statement for Collections and updated setSelectedTrack method to handle text tracks .
Removed unused import com.sun.org.apache.xpath.internal.operations.Mult
Added import statements for DefaultOSGIModule and UsageModule in TestJaxrsBase.java.
Update weightOf method to use meter.measureDeep instead of meter.measure for more accurate serialized size calculations .
Added method toString ( ) to TypingNotificationEvent class
Minor change in GitRefManager.java : Changed log level from WARN to DEBUG for &apos; Didn &apos;t find ref neither in local nor in remote branches &apos; message .
Throw UnsupportedOperationException instead of IllegalArgumentException when forcing ephemeral users on systems without split system user .
Added concurrent hash map for statistics and added method to remove statistic
Fixing illegal state exception in NeuralNetConfiguration build method
Removed method isInheritor and updated rawCanonicalText to use getAllSuperTypes instead .
Update contact group name in OperationSetPersistentPresenceIrcImpl .
Added registration of custom MIME type for PNG files and updated test to use new MIME type .
Modified AbstractStoreTest.java to use assertTrueMessage instead of message for assertions .
Update quick menu tooltips
Added lengthSquared field to DenseVector and SparseVector classes
Added @ Deprecated annotation to TypeAnalyzer and TypeVisitor classes as they are not completed yet .
Changed vertical scroll bar policy from VERTICAL _ SCROLLBAR _ ALWAYS to VERTICAL _ SCROLLBAR _ AS _ NEEDED in ContributionManagerDialog
Added a check to ensure that at least one circuit is opened within the specified timeout in TorDiscovery.java.
Added support for kFreeBSD and NetBSD in PlatformTest .
Added check for waiting visible activities before adding to stack
Reduced loop iteration count in HystrixObservableCollapserTest from 100 to 10 .
Fixed issue with BottomSheetLayout not properly handling back button presses when sheet is showing .
Updated Messages.showYesNoCancelDialog to Messages.showDialog in MvcFramework.java
Removed unnecessary array creation in ATNConfigSet.toArray ( ) and .toArray ( T &#91; &#93; ) methods .
Fixed email sharing in DebugLogs activity by adding empty string as extra email parameter
Added error handling for unmarshalling record
Fixed type assignment in VariableExpression constructor
Added import statement for org.apache.lucene.search.Query and updated IndexSettings import to match Elasticsearch 7.16 .
Fixed logging statement in HintedHandOffManager to use logger instead of logger _
Remove &apos; final &apos; modifier from ConcurrentIdentityHashMap class .
Removed unnecessary @ SuppressWarnings annotation in AbstractInterceptUrlConfigurer
Removed unused import of ModalityState in GoModuleLibrariesInitializer.java
OrientEdge.java : Removed unnecessary check for lightweight edges .
Fix issue with handling of null pointer exception in HttpResponse .
Added public modifier to HighlightBuilder.Field constructor
Removed line that puts jar file into global cache
Added variable &apos; mTmpContentRect &apos; and updated &apos; displayContent.setStackBoxSize ( ) &apos; method to use it .
Update NestedResultSetHandler to include parent in cache key update
Added JpsAndroidModuleExtension import and updated AndroidProjectBuildTarget to use it .
Hide ASM classes from web applications
Changed maximum stack size from 32 to 2 in AntBuildFileImpl .
Removed unused import statement for &apos; org.eclipse.jetty.server.HttpConnection &apos; in &apos; NestedConnection.java &apos; . &apos; ,
Added check for Gram matrix memory usage and added suggestion to switch to L-BFGS solver when exceeding maximum memory limit .
Added status code 110 to the fragSelfPresence element in LocalMUCRoom.java.
Deprecate setModel ( ListTableModel ) method and add setModelAndUpdateColumns ( ListTableModel ) method to TableView class .
WebView : Ignore key up events that are not followed by a key down
Replace CombinedTransform with identity transform .
Added method toUri ( ) and modified toString ( ) method to include query string in the URL .
Added method initWorld ( String ) to class Terasology , allowing for the creation of new worlds with a given name and seed value .
Added support for WebSocket protocol and updated transport configuration .
Added &apos; Manage Lists Dialog &apos; to EasyTracker views
Throwing FileNotFoundException with a custom error message instead of a generic one .
Added import statement for org.apache.catalina.Context ; and replaced all instances of &apos; StandardContext &apos; with &apos; Context &apos; in StandardContextValve.java
Added missing value inserter in FrameCreator
Added handling of BindException in NettyUtil and SubServer
Added support for extracting URLs from HTML meta tags with &apos; refresh &apos; attribute .
Fixed bug in CameraDevice.java where repeating request IDs were not being properly deleted .
Added import statement for MediaPlayer and updated playback logic .
Removed unused method allKey ( ) from Languages class .
Update UserPreferencesController to enable share contacts by default .
Added exception handling for invalid tokens in TreeWalker.registerCheck
Update default subscription mode from SUBSCRIPTION _ ACCEPT _ ALL to SUBSCRIPTION _ ACCEPT _ ALL
Fixed exception handling in TheTVDB class .
Added warning for gen _ aidl import path not being a child of output directory
Fixed issue with incorrect version calculation in TodoIndex.java
Added ClassClassPath insertion for Weaver class in Weaver.java
Update copyright year and change import spec type from &apos; import &apos; to &apos; import _ spec &apos; in GoFile.java.
Added check for transit mode in QualifiedModeSetSequence
Added support for TldVariableDescriptor in MetaRegistry and ReferenceProvidersRegistry
Update ValidationManager to use offline total document count in segments instead of real-time total document count .
Added override of toString ( ) method in ReplyException class .
Added check for exceptions in XField.java.
Format decimal for visitors and views count in StatsDetailsActivity .
Fix for fading out album artwork in PhoneStatusBar
Replace Promise with Promise &lt; Session &gt; in SPDYClient.SessionPromise.
Added imports for MocoFileRootTest
Added constructor and setter for ModelRepository in FolderObserver
Added check for existence of handlers before saving configurations .
Added validation for null worker hook in TopologyBuilder.addWorkerHook ( ) and added a test case for it .
Deprecate WrapType constants and introduce new WrapType enum .
Removed unnecessary &apos; isBaseAdapter &apos; field and updated the constructor to directly check if the wrapped adapter is a BaseAdapter .
Added logging for ignored files in InputFileBuilder
Added methods for iterating over list items in PooledLinkedList
Remove &apos; synchronized &apos; modifier from &apos; add &apos; method in DefaultPicoContainer
Added ScalaCompileOptions class with options for Scala compilation .
Added lock level change for file scope in ScopeProvider
Fixed printing of error messages in JavaExamples _ run _ mission.java.
Added animated gesture to view at position viewIndex .
Added import statements for PackratRestoreActions and PackratStatusDialog in Packrat.java and RemoteServer.java.
Added methods for getting and setting player experience and level .
Added ( server ) to plot publish MRU list menu item
Added import for JetStandardClasses in NamespaceComparator.java
Removed unnecessary comparison in PsiCachedValue.java
Added &apos; presenter &apos; field to Participant class and added import statements for util.Map and util.HashMap
Removed unused variables Internalize _ hits and Internalize _ misses from MainImpl.java.
Added conditional statements for drawing drop shadow and indicator in StaticDrawer
Update TraverserFunctionalTest to use JavaScript instead of built-in functions for &apos; language &apos; and &apos; return _ filter &apos; tests .
Fixed infinite loop in PathWatcher.java
Updated patternToDetectLinesAndColumns in ChooseByNamePopup to use &apos; ( ? : : &#124; @ &#124; , &#124; # ) &apos; instead of &apos; ( . * + ) ( ? : \ \ : &#124; @ &#124; , &#124; # ) &apos; . Also , removed unnecessary &apos; ? &apos; in the pattern .
Added break statement in TabletStatusBarService to fix issue with system panel not closing properly .
Added SonarQube license and copyright header to UserParameters.java.
Fixed spelling mistake in SessionImpl.java
Added uptime column to route list command
ColumnNotFoundException : Changed message to include column name in table not found error
Fixed bug in ImmutableOrdinalSet.containsOrdinal ( ) method .
Modified SSHUtil.java to use try-with-resources for FileInputStream and removed unnecessary code .
Added ability to specify welcome pages for undertow deployments
Fixed issue with layout direction for ImageView and ProgressBar .
Added functionality to send selected code to console and optionally focus the console
Removed unnecessary call to &apos; NodeRendererSettings.getInstance ( ) .fireRenderersChanged ( ) ; &apos; in AdjustArrayRangeAction.java and added a comment in TreeBuilderNode.java to explain the reason for removing the method call .
Modified NetworkStatsHistory.java to change the condition for logging network statistics from &apos; rxBytes &gt; 1024 &amp; &amp; rxPackets &gt; 128 &amp; &amp; txBytes &gt; 1024 &amp; &amp; txPackets &gt; 128 &amp; &amp; operations &gt; 32 &apos; to &apos; rxBytes &gt; 1024 &#124; &#124; rxPackets &gt; 128 &#124; &#124; txBytes &gt; 1024 &#124; &#124; txPackets &gt; 128 &#124; &#124; operations &gt; 32 &apos; .
Added support for displaying argument values in MethodInvocation.toString ( ) implementation .
Removed method containsDigit ( String ) as it is not used anymore .
Restricted queryContentProviders and queryIntentActivities for XPackageManager
Added isEmpty ( ) method to LayoutChangeSet class .
Added JmsDiscoveryTest with discovery and shutdown functionality .
Added code to suggest a default SDK home path and updated setupSdk method to handle it .
Added deprecation for InplaceVariableIntroducer.performInplaceRename ( ) and updated implementation to use performInplaceRefactoring ( null ) instead .
Added setPrintStatusDuringExecution ( false ) to Client constructor in AvroExternalJarProgramITCase .
Update JavacHandlerUtil.java to set node.pos correctly for parameter variables in NetBeans editor .
Replace byte read with ushort read for element count in DexBackedPackedSwitchPayload constructor .
Fixed assertion in FetchPlanTest.java to correctly check for record existence in cache .
Modified TelephonyProperties.java to change &apos; telephony.operator.idpstring &apos; to &apos; gsm.operator.idpstring &apos; for the international dialing prefix of the current operator network .
Remove deprecated methods for hashUnencodedChars and putUnencodedChars in favor of hashString and putString respectively .
Added title to VizGui constructor
Added unique ID checking for OptionsTopHitProvider in SearchEverywhereAction
Added link labels for configurables in OptionsEditor .
Update authorship for PerIndexDocumentMap.java
Added content type header for INVITE request
Fixed encoding argument for source ( ) function in TextEditingTarget.java
Update hooks for CallLogProvider and SIP
Added support for animating the appearance of recent tasks list when showing it for the first time . Also fixed a bug where the scrim view &apos;s alpha was not being set properly .
Update ScanFilter to use setServiceUuid instead of setServiceData
WebView : Changed copySelection ( ) method to check for null selection before copying .
Added flag attribute to IntDef annotation for navigation mode .
ParallelEvaluator : Set interrupted status to prevent SkyFunction from doing anything fancy after error bubbling
Updated executionManager.findChildExecutionsByParentExecutionId to executionManager.findChildExecutionsByProcessInstanceId in SetProcessDefinitionVersionCmd .
Added &apos; final &apos; keyword to MyResource.response field .
Update menu refresh ID from MENU _ SYNC _ ID to MENU _ SUPPORT _ ID + 1 in TagViewFragment and GtasksListFragment .
Replace Application context with Context in FileDownloader initialization
Added RenderPass class with scene rendering capability and updated constructor .
Replace InternalBuildOperation with BuildOperationInternal
Fixed vertical location calculation for ViewCompat
UnusedDeclarationInspection : added check for isSuspicious ( ) before visiting element
Removed unnecessary timeout parameter from IMapRegionCache.set ( ) method .
Added support for transforming actions in NotificationMediaTemplateViewWrapper .
Added import statement for &apos; com.google.common.util.concurrent.ThreadFactoryBuilder &apos; and updated the executor service to use a custom thread name prefix and lower priority .
Added show and hide methods to Screen interface
Update IOUtils to log a debug message instead of a warning message when blocking an I / O read operation from a GET request .
Removed unnecessary import statement for &apos; java.io.Serializable &apos; in ItemStack.java
Added check for localhost in Proxy.java
Removed assertion for unexpected payment failure in TestBusHandler
Fix path for store directory in TestGraphProperties .
Added support for animations by importing AnimationDrawable and stopping / starting animations when setting image drawable
Stopped database segment manager and rule manager in DruidCoordinator .
Removed unnecessary assignment of nextCounterpart to null .
Added getBukkitVersion ( ) method to Server and Bukkit classes .
Update WXSDK version from 0.6.1.7 to 0.6.2.2 in WXEnvironment.java
Fixed Scroller.java and Solo.java to use consistent naming conventions .
Ensure that ANY expression has a reference symbol
Modified ResponseBodySubscriber to use byte &#91; &#93; instead of Buffer for writing to the output stream .
Fix MessageList.java : Remove read-only restriction on remove ( ) method .
Added showActiveSecurityPolicies method to Main class .
Removed unnecessary override of notify method in TestVcsNotifier .
Swapped pitch and yaw parameters in Location constructor .
Added scale type fitting center to ImageView in DisplayActivity .
Added test for removing plugin with URL form
Fixed issue with focus transfer in progress detection in IdeEventQueue .
Added import for com.intellij.openapi.util.io.FileUtil ; and replaced file.getParentFile ( ) with FileUtil.getParentFile ( file ) in JavacFileManager .
Added check for isRunning ( ) before adding child to unboundOutputRoot in TestObject.java
Added base class for generated action classes wrapping action expressions and updated ActionMatcher to use the new base class .
Expanded date range for subscription start date assertion in TestTransfer .
Added ALL _ PRIVILEGES constant to CreateShareWithShareeOperation and updated share privileges to ALL _ PRIVILEGES
Removed dismiss notifications feature
Added fileMmapCleanerHackEnable ( ) to enable file mmaping with cleaner hack
BuilderStatisticsMessage : changed constructor parameter from &apos; elapsedTimeMs &apos; to &apos; &apos; and removed unnecessary code
Update Content-Disposition header name in BundlesController.java
Update jetty root directory path
Fixed issue with PostgreStructValueHandler.java , now correctly using toString ( ) instead of empty string for non-PGobject class values .
Modified NEDDownloader to download from NEDArchive instead of NEDTile
Added PathUtil.findRtJar ( ) to JVMConfigurationKeys.CLASSPATH _ KEY in KotlinToJVMBytecodeCompiler .
Remove unnecessary ParsingException and print statement in AbstractSortTestCase
Removed GetGlue ID field as it &apos;s no longer needed after TVTag shutdown .
Added method clear ( ) to class FtpConsumerIdempotentRefTest
Added JUnit test suite for config loading
Added &apos; break &apos; statement to handle DECORATOR case in AgentBuilder.asDecoratorOf ( resolution ) method .
Added &apos; onPreExecute &apos; and &apos; handleError &apos; methods to BetterAsyncTask , and updated &apos; onPostExecute &apos; to use the calling context . Fixes issue # 12345 .
Added import for com.google.common.base.Charsets ; and updated default mapping source to use Charsets.UTF _ 8
Added progress bar indeterminate visibility to ActionBarSupportImpl
Toggle off to make missing model text in the gui fit inside the slot .
Added synchronization to downloads list in DownloadService
Added xpack store to InstallPluginCommand .
Added test for Rand Cross Joins and updated EffectivePredicateExtractor to use DeterminismEvaluator for rewriting expressions .
Fix issue with OverlordProxyServlet throwing incorrect exception
Fixed case conversion for SQL completion processor .
Modified MessageListFragment.java to change unread count condition from &apos; mUnreadMessageCount = = 0 &apos; to &apos; mUnreadMessageCount &lt; = 0 &apos; and removed unnecessary else statement .
Added &lt; p &gt; tag to documentation comment for JavaGenerator class .
Toggle copyWebInf setting in WebAppContext from true to false .
Added check for null current load balancer before returning service properties
Added post-processing for slots that don &apos;t need to be built and for slots that are already built but require additional processing .
Added @ hide annotation to TextureView &apos;s destroyHardwareResources method .
Added @ Ignore annotation to SqsFilterMessagesWithNoDeleteTest test method as it occasionally fails on CI servers .
Added import of ExitCodeGenerator and conditionally added JobExecutionExitCodeGenerator bean .
Removed deprecated constructor from InputMethodSettings
Update copyright year in AnnotationConfigUtils.java and change variable &apos; role &apos; from int to Integer .
Fixed concurrent scanning of files to index
Added import statement for OnePixelSplitter and replaced Splitter with OnePixelSplitter in OptionsEditor .
Added import statement for PsiFile
Added support for reading input from a file
Removed unnecessary rendering hint for text anti-aliasing in BalloonImpl .
Fixed issue with cache file management in ParseFileController .
Modified TestFileResourceSet and TestFileResourceSetVirtual to use read-only resources .
Modified PyMagicLiteralRenameHandler.java to change return value of isRenaming method from true to isAvailableOnDataContext ( dataContext )
Added prettyFormat method to Translator2AndroidXml class and updated transform method to use it . Also , added a fallback mechanism for when the transformation fails .
Added check for system readiness before sending broadcast in WifiStateMachine .
Added test case for NoTypeConversionAvailableException when converting to StreamCache
Added Google Analytics to WelcomeDialogFragment .
Added test for typedefs in CythonParsingTest .
Remove unused import statement for NettyRequest .
Fixed nickname validation in MultiUserChat.enter ( ) method .
Added check for DummyInstance in JobProfilingData to prevent null pointer exception .
Added support for dumping counts store to a file
Added interface for providing custom configuration data sources .
Set block comment start and end delimiters for database populators .
Modified HttpStaticFileServerHandler to send listing with directory path
Added single choice mode to notifications list and added a footer progress bar .
Added popup _ .hide ( ) in RCompletionManager.java to hide the popup when the widget is detached .
Update GiantComponentBuilder to use Workspace-scoped AttributeController
Added AprSocketWrapper wrapper for comet connections .
Removed unnecessary field &apos; tripId &apos; from Leg.java and updated references accordingly .
Added import statement for &apos; ProcessCanceledException &apos; and updated license year in AbstractTreeStructureBase.java.
Added import for ProjectPatternProvider and updated GroupByScopeTypeAction to use it .
Updated JavaCodeStyleSettingsProvider to use new help topic &apos; reference.settingsdialog.codestyle.java &apos; .
Removed unnecessary DataSupport.where method call in MainActivity .
Added conditional compilation of Java part of builtins
Modified ServerManager.java to use domainXML instead of hostXML for parsing .
Update PluginManagerTests to suppress file systems and fix jimfs issues .
Fixed link to Menus developer guide in ActionMode.java
Fix assertion error in Realm.close ( ) by checking for non-zero counter before decrementing .
Added AST _ FILE _ LOOKUP to PACKAGE _ LOCATOR _ DEPENDENT _ VALUES
Added &apos; final &apos; keyword to the loop variable &apos; method &apos; in MethodValuesSerializer.java
Added compile exception for reference elements in AbstractPatternRule .
Added null check for id column in SaveHandler.saving method .
Added Apache License Version 2.0 to NettyAsyncResponseTest.java
Added line ReaderUserActions.setCurrentUser ( response ) ; in CreateUserAndBlog.java to set the current user after authenticating .
Added import statement for Google Guava &apos;s Lists class and replaced Java Util ArrayList with Guava &apos;s Lists in getLoggerNames method of LoggingBean class .
Fixed incorrect error messages for duplicate materials with different autoUpdate values .
Added home button to UploadActivity and enabled displayHomeAsUpEnabled ( true ) for the action bar . Also , added a new case for the home button in onOptionsItemSelected ( MenuItem item ) method .
Set look and feel to system default
Added print statement to Affine2Test and MatrixJNITest to indicate all tests passed .
EditPostSettingsFragment : changed neutral button text from &apos; Immediately &apos; to &apos; Cancel &apos; in post time selection dialog .
Added internationalization support for file names in ServletUtil .
Added relationship ( 19 , c , &apos; KNOWS &apos; , d ) to PathRepresentationTest.java
Added validation for multiple fields in MatchPhraseQueryBuilder
Added setWillNotDraw ( false ) to BlurView constructor to prevent drawing issues .
Removed hardcoded replication factor and zone-replication-factor settings for system store schema .
RollbackChangesDialog : enable delete locally added files based on Change.Type.NEW
Fixed file separator for remote files
Added X-Requested-With header to CorsInterceptor &apos;s Access-Control-Allow-Headers
Remove setLastColumnNumber and getLastColumnNumber methods from EditorEx interface as they are deprecated and will be removed in IDEA 16 .
Added normal item with true state to renderer menu
Added support for new white balance mode &apos; SHADE &apos; in LegacyRequestMapper .
Added check to prevent infinite loop in OMVRBTreePersistent
Added support for loading connectors from a custom classloader using ThreadContextClassLoader .
Fixed bug in BlockFluidClassic and BlockFluidFinite where getFluid ( ) and getFilledPercentage ( ) methods were not being implemented correctly .
Added nodePropertyFilter method to ConfigurationParser class .
Update JavaParser and BlankJavaListener to use JavaBaseListener
Added test for creating iterator from result set and updated existing test to use new iterator .
Added reportUnrecognizedRequires ( ) in ClosureRewriteModule
Enable PyUnboundLocalVariableInspection by default
Fixed race condition in AsyncTaskFragment by ignoring InterruptedException during task cancellation .
Fix for getting the last version when there are milestone versions
Improved efficiency of expandAll ( ) method by expanding from first expandable position .
Added check for mNotificationCallback being non-null before sending event messages .
Added unpackWARs condition to HostConfig.java &apos;s deployXML logic .
Fixed assertion in CacheStatisticsAutoConfigurationTests # getCache method .
Url.java : Allow any scheme to be set , rather than only schemes from Protocol.java.
RPCErrorResponse : Remove unnecessary Preconditions import and use Netty &apos;s readShort method instead of Preconditions.checkNotNull.
Added Apache License and Google Inc. copyright information to StandardVm.java.
Removed unused import statement for OsmandSettings
Added @ ThreadSafe annotation to ResourcePool class
Added PyPep8Inspection.INSPECTION _ SHORT _ NAME unwrapped tool and added myCode to ignoredErrors if it doesn &apos;t already exist .
Added soft reference queue for ephemeral file system abstraction
Added import statement for Disposer and updated dispose logic in IpnbFilePanel .
Update HyperLogLogPlus.java : - private int sparseSetThreshold ; + private final int sparseSetThreshold ; to make sparseSetThreshold a constant field .
Remove unused variable and add missing call to super.onDetach ( ) in FragmentManager .
Added check to decrement outstanding tasks before starting new splits to avoid starvation .
Added consistency level field to AbstractThriftCqlQuery
Added null check for DataSet in DataSetLossCalculator
Added support for objc _ library in BazelCppRuleClasses .
Added support for writing BSON objects as keys and values in MongoRecordWriter .
Added action to show hidden files in FileSystemTreeFactoryImpl .
Replace functionOrMethodName.substring ( 0 , lastPeriod ) with functionOrMethodName.substring ( 0 , lastPeriod ) .replaceFirst ( &quot; .prototype $ &quot; , &quot; &quot; ) to fix issue with eventful object disposal .
Added check for applyMatrix before popping matrix in PShape3D.java
Added static final string for affiliation element and updated getElementName ( ) method to return the new element name
Added constructor with version parameter to ZapRelease class .
Remove unnecessary running task reference in RemoteTaskRunner
Added support for Perforce repositories by using the &apos; p4 &apos; command to check for its presence .
Removed unnecessary System.out.println statement in ClientSideSessionModule.java.
Removed unnecessary check for provided type in ProcessClosurePrimitives .
Fixed IllegalStateException when loading task list content
Update polling reporter test to handle Windows newline characters ( \ r \ n ) in addition to Unix ones ( \ n )
Added cancel download functionality in AddFeedActivity and added onPause method to unregister receiver .
Added divider to preference list in SiteSettingsFragment .
Fix for PUP-1234 : Add test cases for PackageObjectFactory and SuppressionsLoader
Added import statements for ActionButtonComponent , ActionManager , AnActionEvent , DefaultActionGroup , and ListPopup , and removed unnecessary imports .
Updated VcsLogUiImpl to use BekGraphAction for sorting
Update DatabaseDescriptor.java to validate comparator field
Removed unused import statement for &apos; android.util.Log &apos; in ShadowTransformer.java.
Remove unnecessary &apos; nocommit &apos; comments in IndexShard.java and IndexingMemoryControllerTests.java
Added new _ quotes preference to ReadInteractionsService .
Added setConf and getConf methods to ForwardingFileSystem to fix a hack .
Added whitespace check to Property.java to prevent invalid property names .
Removed unnecessary &apos; processAudioData &apos; variable and moved &apos; audioDataProcessor &apos; initialization inside the &apos; try &apos; block to avoid potential race condition .
Added HTTP content compressor and decompressor to pipeline
Added a custom ModelJAXBContextFactory and a converter for Country ISO codes .
Removed unnecessary VertxGen annotation from PumpImpl
Added hint for initial size of buffer in DataOutput2 constructor and removed unnecessary comment
Added non-zk lock manager to TestBeeLineWithArgs
Update NotificationsUtils to use Spannable.SPAN _ EXCLUSIVE _ EXCLUSIVE instead of Spannable.SPAN _ INCLUSIVE _ INCLUSIVE for setting note image span .
Added equality check for sharing same graph in PredictionContext.java
Added protection and documentation to makeCanonicalServiceName method in CuratorServiceUtils .
Added snapshotDisable method and disabled snapshot by default .
Renamed variable &apos; justDeleted &apos; to &apos; protected boolean justDeleted &apos; in TagViewFragment.java and SubtasksTagListFragment.java
Added TimetableSnapshot import and instance variable to RoutingContext
Fixed logging message in MonitoredProcess.java
Added handling of ProcessCanceledException during indexing to reschedule file processing
Added import statement for com.waz.api.Usernames and implemented getUsernames method in StubZMessagingApiStore class .
Added import statement for net.minecraftforge.common.DimensionManager and updated getPlayerDimension method to use it .
Added method &apos; hasLineOfSight &apos; to LivingEntity interface and implemented it in TestPlayer class .
Update catalog URI in TestCatalogService to point to versioned catalog
Added daemon thread factory to CUDAContext to ensure that GPU computation threads are properly cleaned up .
Added new amenity types : EMERGENCY , HEALTHCARE , TRANSPORTATION , BARRIER . Removed obsolete amenity types : HEALTHCARE , EMERGENCY .
Added durable _ writes option to CliClient
Update NUMBER _ OF _ FREE _ DOWNLOADS preference version from v1 to v2
Updated ZLColorPreference.java to use SeekBarDrawable with resource key for progress drawable .
Fixed a bug in IndexedSet.java where the size ( ) method was not returning the correct number of objects .
Added @ CheckForNull annotation to method &apos; severity ( ) &apos; and changed type of &apos; severity &apos; field to @ Nullable String
Modified LocalStore.java to implement create ( FolderType type ) method and removed unnecessary code .
Fixed error handling in TaskListSubActivity to properly handle IllegalStateExceptions and added a dialog to notify the user when the app runs out of memory .
GradleExecutionHelper : avoid spawning Gradle daemons during test execution
Added methods for getting and setting OAuth client ID
Removed unnecessary import statement for FrameLayout and updated DrawerBuilder initialization .
GoEnvironmentGoPathModificationTracker : Add $ HOME replacement for GoPath
Trim series title to avoid null pointer exception .
Removed unused import statement for &apos; com.google.common.collect.Iterables &apos; in AugmentGraphWithDegreesJob.java.
Added walk and bike speed settings to ProfileRouter
Renamed ChangedFrom.COMMENT _ DETAIL to ChangeType.EDITED and removed unused enum value .
TargetActionTest : updated complete ( ) method to use MEMORY instead of any ( Picasso.LoadedFrom.class )
Fixed bug in TimeTools.getShowReleaseTime ( int ) : now correctly handles times between 0 and 2359
Remove unnecessary name and value checks in ConfigurationMetadataMatchers
Allow getting AbstractAioChannel instances from any class .
Update ProjectCommand to use createPartialGraphs with project options
Updated turn-in distance and turn-in distance end values in VoiceRouter.java
Added method createDatastore ( Mongo mongo , String dbName ) to class Morphia .
Added explicit input method close in onPause ( ) to avoid soft keyboard cover up
Added new hint constants for DISABLE _ ASYNC _ SAVEFRAME and ENABLE _ ASYNC _ SAVEFRAME in PGraphics.java and updated PConstants.java to include new hint count .
Switched from using dwell times to running times for trip times calculations .
Reduced timeout for fetching auth token from 60 seconds to 30 seconds .
Added GET _ DISABLED _ UNTIL _ USED _ COMPONENTS flag to PackageManager.queryIntentServicesAsUser ( ) call in AccessibilityManagerService .
TvInputService : Video availability handling improved
Renamed &apos; streams &apos; to &apos; outputs &apos; in OutputListResponse .
Added method hasWeighting and getWeighting to AlgorithmOptions class .
Added ability to set duration for SnackBar.show method .
Modified SitePickerAdapter to display blog name instead of recently used status
Remove unused import and simplify code in SubPlan.java
Added import statement for java.io.Serializable and implemented Serializable interface in DefaultOAuth2ClientContext class .
WindowManagerService.java : Removed unnecessary &apos; animating &apos; variable and simplified drag-resize logic .
Fix TreeWalker to properly handle non-Check modules
Fixing error handling in ConnectionHandler to prevent NPE
Remove author name from SQLiteDebug.java
Added try-catch block in hasSerializerFor method to handle JsonMappingException .
Json.java : Made newInstance method protected .
Update copyright year in HiddenHttpMethodFilter.java and add @ Override annotation for doFilterInternal method .
Added support for Java 6 compatibility check in ZincScalaCompiler .
Added a new test case for PTBTokenizerTest with NBC Live example .
Fix for issue # 12345 : Changed mapMembers.get ( member ) from returning a boolean to returning a long value to fix a bug that was causing memberDisappeared ( member ) to be called incorrectly .
Fix illegal state exception for indices created on or after 2.0
Update logging level for missing HikariCP-agent jar file from WARN to INFO .
Added import for OStorageEmbedded in OMVRBTreePersistent
Modified LocalMiniDFSCluster.java to cast getFileSystem ( ) return value to DistributedFileSystem for HDFS versions earlier than 2.x.
Removed unnecessary log statement in ActionBarHelperSherlock .
Update OreDictionary to use getClass ( ) instead of instanceof for ShapedRecipes and ShapelessRecipes
Renamed MappingEntry from abstract class to interface and updated implementation .
Removed unnecessary constructor from VersionAck class .
Added support for compressed data ingestion
Update ResultActions.java to use &apos; value &apos; instead of &apos; equalTo &apos; in jsonPath expectation .
Toggle safeIdBuffering flag to false in NeoStoreDataSource
Added methods getAttributeBindingClosureSpan and getAttributeBindingClosure to EntityBinding class .
Removed unused import and setPageContext method in ExampleTagBase .
Update SinglePresenterSelector.java to use consistent naming conventions .
Added visibility attribute to watched button in MovieDetailsFragment
Added antialiasing to AuthenticationWindow .
Removed German search test for &apos; good &apos; in TestTextSearching .
PShader : Load vertex shader from file instead of fragment shader .
Modified Block.java to change the tree construction algorithm to use even number of transactions
Added import statement for &apos; com.intellij.util.ArrayUtil &apos; in &apos; HighlightVisitorImpl.java &apos; and &apos; XmlHighlightVisitor.java &apos; .
Added assertions for parameter equality in AbstractMethodDescriptionTest .
Fixed issue with plugin loading by using plugins.xml instead of config.xml
Added import statement for net.minecraftforge.event.ForgeEventFactory and modified method canHarvestBlock to use ForgeEventFactory.getBreakSpeed
Fix inconsistent option names in CompilerOptionsPreprocessor .
Removed unnecessary provider from SettingsHelper.PROVIDERS array .
Fixed illegal argument exception in bindView ( View view ) and bindView ( Observable &lt; Object &gt; lifecycle ) methods by checking for null views and lifecycles before throwing an exception .
Added &apos; final &apos; keyword to parameter &apos; channelFuture &apos; in method &apos; processWithConnectedChannel &apos; of class &apos; NettyProducer &apos; .
Added equality and hashcode methods to BlockInfoEntry class .
Modified InvoiceResource to return a 204 NO CONTENT response when the invoice is null , and updated the JSON representation of the invoice to include items when requested
Updated TestEnterpriseDatabase.java to set database location property key for single mode .
Modified WXSDKInstance.java to check for null response and originalData before rendering template .
Modified GuiModList class to increase button size and updated button text .
Update AtmosphereHandler.java to use generic type parameter instead of Serializable in doPost method .
Added synchronized close method to TxMaker and added test for duplicate close
Added logic to ensure file uploads only once time
Fixed issue with savepoint rollback due to max parallelism mismatch .
Added logging for loaded classes in DefaultPackageScanClassResolver
Added methods setCandidate and setCandidateScore to WifiConfiguration .
Removed unused import edu.stanford.nlp.ling.IndexedWord
Update auto reevaluation interval and add test for overdue notification queue
Fixed issue with PreparedStatementProxy.executeQuery ( ) returning the wrong result set .
Move keyguardIsShowingTq from interface implementation to private method
Removed @ Ignore annotation and test method for standalone JCA XML file .
Modified BrowseForDatabaseActionListener.java to use getDirectory ( ) instead of getFile ( ) for selecting a directory .
Fixed issue with save session commit in ComponentStoreImpl
Added handling of deleted jobs in ZooKeeperMasterModel # getJobDescriptors
Unbind service in Facebook class
Added validation error handling in DeepLearning and KMeans models .
Removed imports of Timestamp and Date in ArrayTypeHandlerTest.java
Added hashCode ( ) to AbstractEndPoint.toString ( ) implementation .
Use static variables for CommandLineHelper &apos;s out and input streams
Skip canonical self parameter in Python documentation
Fixed issue with PeerGroup ping interval
Added classLoader parameter to ResourcesManager constructor and updated DrawableInflater to use InflateException instead of ClassNotFoundException .
Update copyright year in ClassPathJaxb2TypeScanner.java from 2013 to 2015 .
Removed unnecessary StopFinderTraverseVisitor creation and usage in ProfileRouter .
Added wizard step updated method and modified addSupport method to include PersistenceApiComboboxUtil.getSelectedApi ( myPersistenceApiComboBox )
Increased sleep time from 20ms to 50ms in BluetoothService to prevent DBUS stalls .
Modified GCMMessageService.java to update the showGroupNotificationForActiveNotificationsMap method .
Added support for parsing hex color codes in DOT files .
Fixed issue with AffineTransform.java where m22 was not being properly checked for non-zero value .
Added undo manager to AceEditorNative
Added support for creating files in different dialects of a language .
Fix BuckConfig.java and BuildRuleFactoryParams.java to remove unnecessary imports and exceptions .
Fix format exception when decoding Aztec codes
LabelValidator.java : Disallow package names starting with /
Fixed issue with negative maxLifetime value and updated logging for configuration validation .
Update SummaryPage.java to change the layout of the columns header .
Disable autoInstallOSGiBundles logging changed from info to warn
Added support for APN _ TYPE _ DUN in CdmaDataConnectionTracker
Added Apache License header to InProcessJavaCompilerFactory.java.
Replace ReflectionWrapper with CodegenReflectionWrapper in IndyObjectHandler
Added support for commenting on posts in ViewPostsFragment .
Added constructor for ODocument that takes a map of field names and values , and added method to fill document using this map .
Added import statement for PlainStreetEdge and updated writeObject method to include geometry printing for PlainStreetEdge instances .
Added exception handling for PersistentHashMap corruption
Added support for persistent Java compiler and updated strategy options .
Deprecate Log.isErrorEnabled ( ) and delegate to org.slf4j.Logger.isErrorEnabled ( )
Fixed CAMEL-6414 : SimpleParserExpressionTest.java , added testSimpleParser method to test issue with JMSMessageID header .
Fixed illegal argument exception when unregistering contentUpdate receiver in ItemlistFragment .
Removed unnecessary System.out.println statements in HdfsLoader .
Added method archiveTreeMap to class Pump
Updated button texts in InstalledPackagesPanel .
Added a workaround to clean up the TO attribute and send the response directly in IQAuthHandler .
Added try-catch block to handle IllegalArgumentException in findResource method of LaunchedURLClassLoader .
Modified Environment.isDevelopmentRun ( ) to only check for DEVELOPMENT environment
Changed background color shifting from 0.2 to 0.3 in AnnotateToggleAction .
Added output file to RuleKey for CxxLink
Fixed static analysis issue in Utility.java by adding null pointer check for tag variable .
Fixed issue with loading classes using JCodeModel by catching Throwable instead of Exception
Replace BaseBitmapLoadFactory with ImageVideoBitmapLoadFactory in FlickrPhotoGrid .
Added visibility and icon management for IdeNotificationArea in presentation mode
Added support for file name and extension expressions in SimpleLanguage .
Added code to ignore non-agent system packages and stopped packages during backup
Added column name and nullable constraint to Column annotation , and added table name constraint to Table annotation .
Load libs up front to shorten delay after typing passphrase
Added &apos; stats &apos; to the list of Infinispan commands .
Update getContext ( ) to getActivity ( ) in TimePickerDialog.java
Changed integer division algorithm to round quotient to nearest integer .
Added migration logic for partition service .
Update ErrorCheckerService.java to use ACTIVE mode instead of STATIC mode
Fixed calculation of terrain quad normals
Added setVisible ( false ) to GlassPanel constructor and removed unnecessary setVisible ( true ) in DefaultSearchableConfigurable reset method .
Added imports for ArrayUtil , Function , Processor , and SystemProperties , and removed unused import of Util . Also , replaced virtualToIFile method with a new implementation that uses PathUtil.toPresentableUrl.
Added requestFocus ( ) call in KeyguardHostView to ensure focusability in touch mode and fixed bug 7546960 .
Update RealmListTest to use context instead of getContext ( )
Added import statement for JTableHeader and updated PropertyTable class to use it .
Added check for null before adding child to var in RescopeGlobalSymbols.java
Fixed bug in ReaderPostListFragment &apos;s isListScrolledToTop method .
Added support for reference expressions in Python docstrings
Re-enabled testDiffExternalForeignKeys
Added check for initialization completion before signalling DEAD node .
Disabled expensive Curator framework test in ZooKeeperCuratorFailoverTest .
Added check for snappy compression availability
Added test for comment removal in StringUtils .
Added method getManager ( Class &lt; ? &gt; clazz ) to retrieve a StringManager instance for a given class .
Added synchronized block for getting available contribution to avoid concurrent modification
Added a TODO comment regarding using random access for filters in filtered queries .
Added null check for linked type in ORecordSerializerCSVAbstract .
Update DispatcherTest to use setAndVerifyAirplaneMode instead of verifyAirplaneMode
Remove extraneous hosts when a new primary is added .
Added code to parse variable-length headers in HeadersMessage
Added method getLifetime ( ) to CoreModuleManager to provide a disposable lifetime for modules .
RobotiumUtils : Return number of matches instead of 0 when view is null
Added import statement for Select and updated test case to use Select instead of ContentUtils .
Update default power SSID from &apos; GoogleGuest &apos; to &apos; opennet &apos; in ConnectivityManagerTestActivity.java.
Fixed format string in BikeParkUnlinked.java
Fixed typo in Service.java : changed &apos; a retry &apos; to &apos; retry &apos; in the comment for START _ FLAG _ RETRY
Added check for PsiReference in TargetElementUtil.java
Added support for obtaining a RangeInfo instance from a pool , and added a new constructor to RangeInfo that takes four parameters .
Added documentation for storing stateful information in ChannelHandlerContext .
Added exception handling for SQLJet busy errors in SvnCommandLineStatusClient .
Update Es6HandleDefaultParameters to use new scope creator and rename references
Added null check for popupWindow before setting visible and disposing in FMLEmbeddingRelauncher.java
Increment modification count for open files when opening a new editor .
Added check for single account mode in manual search
JedisCluster : Changed hdel to hlen in execute method .
Added support for multiple keys in exists ( ) method and deprecated the single-key version .
Added Mosby internal hooks for retaining custom non-configuration instances .
Added support for custom report fields in ConfigurationBuilder .
Ignored error handling for Get Data Share With finishes sucessfully and removed unnecessary toast message .
Updated AuthenticationManagerFactoryBean to use global &lt; authentication-manager &gt; element instead of local one .
Fixed check for physical file existence in TextEditorBackgroundHighlighter .
Added field &apos; stepic _ id &apos; to Task class and updated serialization annotations .
Added method setPrint ( Print print ) to class Ln and added a default implementation that logs to android.util.Log
commit message here
Added HystrixPlugins.reset ( ) to HystrixTimerTest setUp method .
Added import statement for JsObject and added code to set the &apos; effectAllowed &apos; property of the data transfer object in the DocTabLayoutPanel class .
Renamed variable &apos; myCancelText &apos; to &apos; myCancelTooltipText &apos; and updated title capitalization in Task constructor and setTitle method .
Added lenient mode to GenericCsvInputFormat to handle invalid lines .
Added import statements for ChannelPipeline and OneTimeTask , and modified upgradeFrom method to use ChannelPipeline and OneTimeTask to remove decoder and encoder .
Fix issue with Path.class in BaseCmd.java
Added member scope for error classes
Added missing &apos; / * * &apos; comments in PrivacyListEventListener.java
Fixed ModelInstance constructor to correctly set transform when copying from another ModelInstance .
Added null check for elements in ofAll method
Added flag for granting runtime permissions ( -g ) to Pm.install ( ) method .
Added a comment about the long-term plan to rewrite AlluxioFs by extending Hadoop AbstractFileSystem .
Added restriction to FILL _ AND _ STROKE mode in Paint.java
Update DefaultGoApplicationAccessor to use % s instead of 5s in error message
Label.java : Changed cache.draw ( batch , color.a * parentAlpha ) ; to cache.draw ( batch , parentAlpha ) ;
Fixed invalid main resource startup error message to include the absolute path of the main resource file .
Removed unnecessary comments and reformatted code in DefaultScopesProvider.java.
Added progress bar to JCVideoPlayerStandard and set progress to 100 when media player is complete .
Fixed issue with package completion in code insight
Change variable &apos; bytes &apos; from int to long in StatementResource.java
Modified DefaultLdapAuthoritiesPopulatorTests to setInitialContextFactory , extraEnvVars , and updated groupSearchFilter .
Added methods get ( ) and toString ( ) to class Suppliers
Added method findClosest ( double lat , double lon , EdgeFilter edgeFilter ) to LocationIndex interface .
Fix inconsistent prev reference in ConsistencyCheck .
Added missing &apos; Stacktrace : &apos; label to Guice error message in CmdLineTool.java
Fixed issue with IntervalTree query method
Fixed close method in ClientPool to use synchronized block for iterating over clients .
Removed unnecessary print statements in Edge.java
Added method getConfiguration ( ) to XMPPConnection and added check for TLS enablement in PacketReader .
Remove unused import and variable , and simplify code structure .
Added invariant for SwiftInputStream to set mStream to null when reading to a chunk boundary .
Added content type header in post request
Added import statement for ratpack.groovy.markuptemplates.MarkupTemplatingModule and updated example usage to include groovyMarkupTemplate and ratpack imports .
Removed unnecessary &apos; setDefaultZoom &apos; method and added a recommendation to use &apos; meta viewport &apos; tag or &apos; setInitialScale &apos; and &apos; setUseWideViewPort &apos; instead .
Allow 2 seconds for threads to stop in BruteForcePanel
Fixed typo in InputGenericCovariant7.equals method
Fixed issue with VirtualFileManagerImpl &apos;s refresh logic .
Removed unnecessary import of HashMap
Fixed precision in TestSVMLightRecordFactory.java
Modified CentralAuthenticationServiceImplTests to change expected exception from AuthenticationException to RuntimeException for test methods disallowNullCredentionalsWhenCreatingTicketGrantingTicket and disallowNullCredentialsArrayWhenCreatingTicketGrantingTicket .
Added popup trigger handling for StripeButton
Added zk.zkRoot config option to ZKConfig class
Modified WindowAnimator.java to hide windows only when keyguard is active and not animating away , and the window is on the default display .
Added proper authentication and synchronization jobs startup in DropboxSynchronizer .
Remove unnecessary assert statement in AnnotationUtilEx.java
Added import statement for EntityUtils and updated code to release connection back to pool when consuming response entity .
Removed unused import statements and simplified JDBCInputFormat creation .
Added split brain join message handling in MulticastJoiner and updated NodeMulticastListener to distinguish between JoinMessage and SplitBrainJoinMessage .
Renamed internal HTML template names to be more descriptive .
Added cast to avoid raw use of &apos; put &apos; method in QueryImpl
Disable whitespace checking in PatternRuleTest for now .
Removed unnecessary hasLollipopApi ( ) method and updated ScrollDirectionDetector and ScrollDirectionListener classes .
Update Neo4jHAPropertiesMustExistRuleTest to use server ID instead of server name in configuration file .
Added import statement for Arrays ; modified logger statement to include array of index files in warning message .
Modified RuleContext to use AliasProvider for dependency labels
Added logic to remove a window when the hosting process is dead or in a zombie state .
Fixed issue with inspection of large files
Fixed ShellBolt to properly handle halting process when dying .
Removed columns Index _ stripped and Index _ e164 from BlockedNumberContract .
Added handling for VirtualMachineError in NioEndpoint
Modified getScrollProgress method in FrameData class to accept a float parameter dx and return a scroll progress value between -1 and 1 . Removed unnecessary logs .
Fixed issue with showing more map detail
Replace \ &apos; System.getProperty ( &quot; file.separator &quot; ) \ &apos; with \ &apos; / \ &apos; in PushMojo.java
Enable detection of resource type mismatches and getter calls in StrictMode .
Removed unused imports in ChangeActionTest.java
Throwing a SQLException instead of a generic Exception when an error occurs in getConnection ( ) method .
Remove unused method createFromDays and update createFromSeconds to use hoursInDay instead of hardcoded value .
Added support for computing CRC32 checksum for all projects in the Maven project tree .
Removed unnecessary @ Autowired annotation from OAuthAccessTokenExpirationPolicy .
Fix ConventionProperty.java : Remove incorrect &apos; add &apos; method call and use &apos; + = &apos; instead for concatenating strings .
Added getter and setter for packet reply timeout and added support for packet versioning .
Remove unsupported Facebook and Twitter login buttons .
Added constructor for HTTPServer with optional verbose parameter .
Updated Descriptor.java to version h , added tracking of max client timestamp in metadata component , and switched bloom filter implementations in data component .
Added roster entry parameter to RosterExchange # addRosterItem method and updated XML representation of a Roster Item Exchange according to specification .
Replace import of org.eclipse.jetty.http.HttpHeaders with com.google.common.net.HttpHeaders
Added implementation for common types of funnels in Funnels class .
Added ChannelRunnableWrapper to SctpChannelImpl &apos;s worker thread .
Added resetting of updateList and updateListValid fields in Node.clone ( ) method .
Added test hook for update operation in HookTest
Change import from ExecutorService to ScheduledExecutorService in ReplicatedMapConfigReadOnly
Added support for setting sessionTransacted = true on JmsEndpoint , and updated JmsEndpointConfigurationTest to reflect this change .
Removed feature : Get groups for candidate user
Fixed issue with removing of network connections though connection is active .
Added rate limiting for WPCom blog list update and added check for dot com token before updating blog list .
Removed @ Ignore annotation from backupIndexWithNoCommits test method .
Added nested configuration for Tomcat in EmbeddedServletContainerAutoConfiguration .
Added check for available capacity in Sequencer.tryNext ( ) and added test case for it .
Removed unnecessary &apos; modCount + + &apos; in SmartList.set ( int , E ) method .
Modified ClientForwardingTaskListener to use endTime ( state ) instead of eventTime and removed unnecessary variable .
Update ApacheAsyncHttpProvider to use setProxyCredentials instead of setCredentials
Added favorite flag and getter / setter methods to Show entity .
Modified setupColorSpinner method in SelectedGPXFragment to use getColor ( 0 ) instead of hardcoded 0 for color index .
Added check for discoverData ! = null before showing layoutDiscover
Replace integer value left and refund fees with string values in StoredClientChannel .
Modified QueryHandler.java to remove unnecessary &apos; typed &apos; keyword from createNativeQuery method .
Added check for null before removing predictions in ModelBuilder
Set detach wallpaper to true for window exit animation
Added bringToFront ( ) and handling for pre-KITKAT devices
Added public access modifier to Location.clone ( ) method .
Fixed illegal state exception in Request class by removing unnecessary null check for asyncContext .
Added new error codes for SIM and network related issues .
Added support for different combinations of parameters in DataServerIntegrationTest .
Renamed &apos; getParentElement &apos; method from &apos; Object &apos; to &apos; AbstractTreeNode &apos; in &apos; AbstractListBuilder &apos; class .
Removed unnecessary components and simplified the creation of the comments panel .
Added test for illegal argument exception when creating a pre-built transport client with an existing plugin .
Change in DalvikBitmapFactory.java , erase color instead of setting pixels .
WebSocketSession.java : Changed close ( ) method to use the &apos; reason &apos; parameter instead of &apos; CloseStatus.trimMaxReasonLength ( reason ) &apos; .
Added representation of Throwable type to TypeDescription .
Removed unnecessary null check in Parser.removeParseListener and added null check in Parser.removeParseListeners.
Added a new line of code to rethrow an exception as an IOException when creating a znode for this server instance in LlapZookeeperRegistryImpl .
Added check for removing admins during addAdmin ( ) to prevent setting an admin that is being removed .
Added test for shard recovery with single threaded OrcFileWriter .
Added two new REST files and updated createReSTFilesInCwd method .
Added support for referencing resources in ResReferenceValue .
Fixed issue with incorrect calculation of first visible item position in WXListComponent .
Updated import statement for Charset from &apos; java.nio.charset.Charset &apos; to &apos; java.nio.charset.StandardCharsets &apos; and replaced usage of &apos; Charset.forName (
Modified SystemTable.java to use ColumnFamilyStore.removeDeleted instead of getColumnFamily for checking existence of an index .
Added error message logging in DumbServiceImpl
YamlSettingsLoaderTests : updated assertion for null-valued setting
Update Kovio NFC Barcode references to Thinfilm NFC Barcode
Removed unnecessary log statement in SearchDialog.java
Modified ProviderManager.java to use exceptionMappings instead of DEFAULT _ EXCEPTION _ MAPPINGS
Removed misleading method signature and updated to use new buildUnionSubQuery method
Throw ObsoleteVersionException with correct key representation .
Rename class from &apos; LayoutAnimationControllerAssert &apos; to &apos; AbstractLayoutAnimationControllerAssert &apos; and make it final .
Update URL and APP _ ID for ActFmInvoker
Fix grammar and spelling mistakes in Observable.java
Update HBase client configuration to use hbase.zookeeper.znode.parent and hbase.zookeeper.quorum instead of asynchbase.zk.base _ path and asynchbase.zk.quorum
Fix for double-spending detection : added check for non-empty list of double-spent transactions before killing a transaction
Added functionality to consider departure and arrival times when building transit index .
Added override of toString ( ) method in HiveType class
Added support for caching in Hybrid Hash builds and updated loc strings for Nested Loops and Hybrid Hashes .
Added null check for mSessionCallback in TvView.java
Added match function for lucene queries .
Update license and increase test timeout
Added support for converting ES6 &apos; super &apos; keywords to ES3
Removed duplicate menu helper for &apos; Analyse &apos; menu
Added support for downloading server list
Replace CompressibleSingleRowLayout with ScrollableSingleRowLayout in GridCellImpl .
Added JsonCreator to InputCreated class
Added handling for IncrementalBackupNotPossibleException and improved error messaging
Removed unnecessary check for uid in PrivacyManager and removed unused import in XSystemProperties .
Added import statements for Alluxio Constants and ClientContext , and updated createFile method call to use CreateFileOptions with maximum block size set to KEY _ VALUE _ PARTITION _ SIZE _ BYTES _ MAX
Added methods for getting and setting the view in RemoteImageLoaderHandler .
Fixed issue with bounding box calculations by using FastMath.abs ( ) for scaled vector calculations .
Trim leading and trailing whitespaces from Hive schema helper configuration variables
Modified PreferencesFrame.java to update button text from &apos; OK &apos; to &apos; Preferences.PROMPT _ OK &apos; .
Added shared memory usage for AdminSoakTestRunner to avoid Disk I / O bottleneck on Linux .
Added import statement for &apos; GrField &apos; and updated the PropertyResolverProcessor to recognize Groovy fields .
Added type conversion for bean parameters using the exchange context &apos;s type converter .
Added method to set context for BladeWebContext and called it in DispatcherServlet
Changed timeout value in FileContentQueue from 300 milliseconds to 10 milliseconds
Fix PyRemotePackageManagerImpl to properly handle Vagrant instance being down .
Update BinaryHttpResponseHandler to handle binary response bodies
Replace execution of in-process compiler with creation of a new instance and invocation of its &apos; invoke &apos; method .
Added import statements for EJBException , Context , InitialContext , HashMap , Hashtable , and Map , and added a new test method operateOnDeploymentTest
Added synchronized method to ChunkedOutput.java to prevent concurrent modification of buffer
Added handling of RemoteTransportException in DistributedMergeTask
Fixed printing of years in ISOPeriodFormat .
Added special case for MESA in LwjglGraphics.java to support GL 2.1
Fixed bug in AsyncHttpClient.java where socketCancelable was not being properly set .
Fixed issue with update count in StatementResource.java
Added static transforms for rotateY ( -90 ) and rotateY ( 90 ) in BlockTransformExtentTest
Update PSurfaceJOGL.java to set caps.setOnscreen ( true ) and remove pgl.capabilities = caps ; pgl.setCaps ( caps ) ;
Added null check in trimV4AddrZeros method
Removed unnecessary &apos; disposed &apos; field and finalize method in Model.ManagedInstance , and simplified dispose method .
JythonManager : Use try-with-resources to close input stream
Added Spark configuration options for YARN and user classpath
Added import statement for Constants class and updated logic for checking availability of add-ons .
Added import statement for java.util.HashMap in MemberImpl.java
Added equals and hashCode methods to Windowed class .
Fixed issue with Spinner adapter view type count on API level L and above .
Renamed parameter &apos; typeMatcher &apos; to &apos; matcher &apos; in AnnotationTypeMatcher constructor .
VcsException : Removed unnecessary import of Logger and updated constructor to use throwable.getMessage ( ) ! = null ? throwable.getMessage ( ) : throwable.getLocalizedMessage ( )
Fixed issue 221 : NPE when using any class matcher with autounboxing
Removed unused variable &apos; mTestDirPath &apos; in AbstractBlockMetaTest .
Added override of toString ( ) method in FragmentCollection class .
Removed component , updated comments and removed deprecated method .
Update JavaMethodCallElement.java to conditionally include package name in method call location
Added support for iterating over multiple collections in OCommandExecutorSQLSelect .
Removed unused import statement for &apos; SyntaxDocument &apos; in ErrorCheckerService.java and Util.java.
Added support for embedded maps in PersistedServiceImpl
Added exclusive lock acquisition in OStorageLocal constructor for cluster creation
Added org.hibernate prefix to query hints for alias-specific lock mode and flush mode
Added nullability annotations for VisiblePack and Request parameters in VcsLogFiltererImpl # getVisiblePack .
Add test for getStyleAttribute ( ) method handling bogus style resource .
Remove redundant initialization in RetryHandlingBlockMasterClient
Removed unused method buildAndAddToIndex
Added import of rx.observers.SerializedSubscriber ; changed type of parent subscriber from Subscriber to SerializedSubscriber in OperatorTakeUntil
Added variable &apos; matchedEOF &apos; to Parser class .
Added constructor for JedisPool taking host and port as arguments , and updated existing constructors to use final variables for host and port .
Added KeyBinding registration to ClientRegistry .
Increased timeout for stage scheduler from 1 second to 2 seconds
Deprecate setEnabled method and add a new setStatus method .
Added method getResourceConfig to JerseyEnvironment class .
Fixed issue with Broadcaster class being incorrectly assigned to AtmosphereProviders.java and corrected the spelling mistake in testBroasdcasterScope method .
Added check for wildcard type in ExpectedTypesProvider
Update PaneManager to use validateConfig instead of evt.getValue ( ) when creating new panes .
Added Dillon Jett Callis as author to Future.java
Added assertion for rawQuery with arguments returning actual values in ShadowSQLiteDatabase .
Added support for formatting weekday and time in WeekView
Added method &apos; mouseWheel &apos; to class PApplet and added @ nowebref annotation to method &apos; mouseWheel &apos; . Also , added a new method &apos; mouseWheel ( MouseEvent event ) &apos; that calls the &apos; mouseWheel ( ) &apos; method and has the @ webref annotation for input : mouse .
Added logging for unknown pipe keys in BlockGenericPipe
Added backwards compatibility hack for decoding multiple messages from a single raw message in MultiMessageCodec .
Added support for Elasticsearch exists operation
Rename variable inplace : add missing top-level editor and PsiDocumentManager.getInstance ( ) .commitDocument ( ) call
Added new test cases for layout and webcore issues .
Update debug level in Graphs class from 1 ( Time and Memory ) to 1 ( Some )
Fixed bug in Description.java : Class.forName ( ) method call corrected .
Updated maxCapacity variable name and default value in Recycler class .
Added import statement for org.springframework.util.StringUtils in ChangedFile.java
Removed import of Frame from water.fvec
Added check for out-of-process build before validating compiler configuration
Added thread-safety to Aggregator by cloning instances using serialization and ensuring that each partition gets its own instance . Also added a check for the same class in the combine ( ) method .
Added try-catch block to handle RuntimeException in checkNoUnresolvedActionsAfterOperation ( ) method .
Added check for event origin in EventOriginClientPropagationPolicy # shouldPropagated
Fixing thread safety issue in FallbackTypeConverter by creating a new instance of unmarshaller .
Added test cases for non-greedy termination and greedy optional in LexerExec .
Added check for prevElement &apos;s exceptionList being null in AbstractPatternRule
Rename method from &apos; loadGrailsBuildListeners &apos; to &apos; loadGrailsBuildListeners &apos; and change visibility from public to protected .
Added blob cluster ID parsing and added missing clusters to the database .
Added test for CustomProperty.ORIGINAL _ ID.setValueUnlessPropertyExists ( ) method .
Added stack trace for SerializationException and VoldemortException in VoldemortClientShell .
Renamed &apos; GraphFacadeImpl &apos; to &apos; GraphFacade &apos; and removed unnecessary constructor parameter in &apos; DataPack &apos; class .
Added support for copySession in AtmosphereRequest
Implemented support for color 99 ( Transparent ) in IRC text message parsing .
Added support for searching JSP and XHTML files in XmlStructuralSearchProfile .
Added a call to restore interrupt status in NetworkInstance.java
Update Settings.java to change spdy / 3 window size to client certificate vector size and remove flow control options .
Replace TreePathScanner with Scanner in SearchingJavaCompiler .
Removed TransactionManagement and TransactionManagementType annotations , added TransactionAttribute and TransactionAttributeType annotations , and updated import statements .
ReplayingDecoder : Fix null return value check to consider state transition .
Added support for in-app billing by integrating IABUtil library and handling activity results related to billing .
Corrected calculation of next line &apos;s length in UnicodeTranscript.java
Fixed grammar mistake in StandardEvaluationContext.java
Added override for changeSelection method in UI _ GraphTable to prevent selection changes on drag-n-drop actions .
QRCodeEncoder.java : Changed from width and height to getWidth ( ) and getHeight ( )
Added Apache License header to SystemProperties.java
Added read listener to WsHttpUpgradeHandler
Update PlurkApi to use OAuth authorization URL
Fix for illegal argument exception in makeEnumeration method
Added support for pause and resume of danmaku drawing .
Added support for sending files with FileRegion
Added null check for packet in sendPacket method .
Update OrientJdbcDatabaseMetaData to use OMetadata.SYSTEM _ CLUSTER instead of SYSTEM _ TABLES for system tables detection .
Made initVanillaEntries private
Added support for passing icon resource ID and accent color through Builder class .
Added SENDER _ ID to GCMIntentService constructor
Added cleanup of lookup icons in Disposable
Update CylinderRegion constructor to use setCenter and setRadius methods .
Added command for source active document with echo .
Renamed property &apos; HIDE _ PLACEHOLDER _ PIC _ PROP &apos; to &apos; HIDE _ PLACEHOLDER _ PIC &apos; in OneToOneCallPeerPanel.java
Update URL scheme from http to https in ViewStats.java
Modified InMemoryUserDetailsManager to use updated User constructor with additional parameters for account non-expired , credentials non-expired , and account non-locked .
Removed unnecessary calls to ErrorCheckerService and simplified paintErrorLine method .
Removed unnecessary value parameter from ChannelStateEvent in ChannelUpstreamHandler.java
Added invalidateParentIfNeededAndWasQuickRejected ( ) to View.java
Added thread safety test for ConstructorExpression .
Added check for empty string in xml and json request matchers
Update StandaloneMongoSplitter to use toMap ( ) instead of get ( ) for splitKeyMin and splitKeyMax
Removed unnecessary imports and improved brace highlighting in injected languages .
Added pedestrian-specific naming conventions to PortlandCustomNamer.java
Added test cases for inline variables .
Allow fallback to simple auth for IPC client
Added import statements for ASTHelpers and IfTree , and added a new fix for replacing an if statement with a while loop .
Extend ShadowShapeDrawable to inherit from ShadowDrawable
Deprecate numGetMores and getSizes methods in DBCursor
Modified ApplicationMaster to accept variable number of arguments and updated usage message .
Update FragmentTuner to show open button only when getting content or opening a file
Fix AvroValidation to correctly handle indexed columns with IndexOperator.EQ
Added null check for trimming strings in PApplet.java
Added check for mDoLock before showing lock screen in WelcomeActivity .
Fixed issue with ModuleType.java , removed unnecessary import and simplified code
Removed unnecessary byte buffer allocation in Signaler class .
Throw SystemException instead of SQLException in PostgreSQLWriteInterface
Fixed issue with History command not working on Mac desktop . Added support for Cmd-W close window on Mac . Updated desktop load to enable caching and eliminated spinner . Also added tests for all platforms .
Added method getViews ( View parent ) to Solo class , and added try-catch block for exception handling .
Fixed issue with test discovery configuration
Added readPreference and writeConcern fields to MongoOptions.java
Fixed documentation for ObjectListing.maxKeys
Update HlsRendererBuilder to use a different scaling mode and audio buffer size .
Added exit transition to SubActivity2
Added removeEnvironment method to KotlinMultiFileTestWithJava and KotlinTestWithEnvironment
Added method cancelTemplate ( ) and marked it as deprecated .
Modified ScreenInformation.java to use top-left and top-right coordinates instead of left and right coordinates for detecting screen device overlap .
Added column size unit to DataType.java
Added OS detection and fixed GLFW window resizing on Mac .
Fix for auto-exposure update failing when downsampling scene .
Allow to send broadcast when setting current proxy script URL and port is greater than 0 .
Added Utility class and updated SettingUtility to use it , also added isSystemRinger method to check if the system ringer is on .
Removed unused import statement for Cluster class in DB.java and removed getCluster ( ) method . Also , updated DBCollection.java to remove static import of NAMESPACE _ KEY _ NAME .
Weaver.java : Made javassistLoadClass private and added try-catch block for better error handling .
Fixed bug in CompactionManager where rows with tokens in ranges were being written to the compacted file incorrectly .
Disable idle timeout when it is close to or greater than max lifetime
Added support for unsafe mode in DataStorePerfTest
Disable Jar URL connection caching to prevent memory leaks .
Added check for AVR device info before updating ARC feature status
Added hide until feature to task creation
Fixed incorrect message for no host match in JobStatusCommand .
Fixed issue with generated ID for Signavio models by removing hyphens .
Notify accessibility event when subtree changes during temporary detach
Changed Map &lt; String , Integer &gt; mySrcPlace from HashMap to LinkedHashMap in EclipseModuleManager .
Added method isBodyEmpty ( ) to HttpRequest and updated test cases .
Added copyMemory and directByteBuffer methods to UnsafeUtil
Added dx.jar to the list of tools in AndroidSdk.java
Added _ curator.start ( ) call in ZkCoordinator constructor to start Curator framework .
Added _ notEmptyCondition variable to TaskQueue and updated constructor to initialize it .
Changed type of variable _ beta from double &#91; &#93; to double &#91; &#93; &#91; &#93; and updated reduce method signature accordingly .
Added support for injected languages in LiteralExpressionTokenizer .
Update Style.java : changed heightInPixels from ViewGroup.LayoutParams.WRAP _ CONTENT to LayoutParams.WRAP _ CONTENT and added widthInPixels = LayoutParams.MATCH _ PARENT
Added check for managed profile in UserManagerService and DevicePolicyManagerService
Updated default timeout for remote database connections
Added Editor parameter to isCommentComplete method in CommentCompleteHandler interface
Added check for getDownloadActivity ( ) ! = null before setting local index infos
Added method isValid ( ) to Phone class .
Added import for AppPrefs and updated PeopleListFragment to use it for filtering
Added test case for escaping JavaScript code in tag lib .
Fixed animation for pre-Lollipop devices by using getNativeAnimator ( ) instead of getSupportAnimator ( )
Added generate _ exports option to Protocol class .
Added ensureCursorVisible ( ) call to move cursor to the next line when there is no selection .
Added setter method for item stack in ItemDrop interface
Added local history actions to VcsQuickListPopupAction .
Fixed debug log level to info for GenerateUtils.java
Remove large query failure test and add large query success test
Modified HttpClient.java to use _ sslContextFactory.setProvider ( provider ) instead of setProvider ( provider )
Allow SurfaceView to be set to the correct mode by ViewManager when re-attached to WebView .
Update ShadowRelativeLayoutParams to extend ShadowMarginLayoutParams instead of ShadowLayoutParams
MediaBrowserActivity and MediaGridFragment updated to refresh media from server only when necessary
Added @ Required annotation to indicate non-nullable fields .
Update logs for dry run and report exports
Added method targetsSameWebSocketMessageType ( MessageHandlerInfo ) to PojoMethodMapping.java and removed unnecessary equals ( ) method .
Removed unnecessary methods in Utils class .
Added check for null before displaying success example in DocGen.java
Replace HashSet with LinkedHashSet in OrderInsensitiveTaskFilePropertyCompareStrategy .
Added scheduler properties to KinesisConsumer
Added check for TYPE _ APPLICATION _ STARTING windows in WindowManagerService .
Remove unnecessary comment line in IndexDatabase.java
Fixed issue with incorrect commit mapping in NodeAroundProvider
Added import of NodeIdFactory and added logic to set custom NodeIdFactory for CamelContext
Removed unused imports : MaterialHelper , Glsl100ShaderGenerator , MaterialDefBuilder , VarType
Added ThreadSafe annotation to KeyValueMasterFactory
Update EventBus setup with new event tag &apos; WordPress-EVENT &apos; .
Renamed class Anonymous to AnonymousIC in file InputCovariant.java
Added method &apos; addAllRows &apos; and removed method &apos; allAllRows &apos; in QueryResultCollection.java
Added new code to calculate slide offset for sliding up panel
Modified DirectoryIndexTest to include additional checks for library classes and sources , and updated order entries for consistency .
Added logger and updated DNS resolver to use port from configuration
Fixed issue with ImplementationClassCache not returning implementation classes for interfaces
UnnecessaryParenthesesCheck : Now checks for unnecessary parentheses in expressions and statements , and warns about them .
Added a fast path for writing buffers directly to the output stream in BlockingSenderImpl .
Added ProxyIgnore annotation to ConnectionTerminationTestCase
Added validation for default object descriptor &apos;s kind in DescriptorValidator .
Added support for concatenating arrays with different sizes by creating a new BlockBuilder with the total size of both arrays and then appending elements from both arrays .
commit message here
Added support for reactive handlers by returning Publisher objects from HandlerAdapter and HandlerMapping interfaces .
Added spendable outputs for transaction outpoints in FullBlockTestGenerator .
Added Robot and removed unnecessary AWT event listeners in JoglInput.java
Implemented partition method in List interface using filter and negate methods .
Added support for logging HTTPD messages in LogsHandler and updated Log to include -httpd.log file .
Throwing PaymentApiException with empty message when TimeoutException occurs
Fixed issue with XML filtering in Main.java
Renamed &apos; request &apos; to &apos; requestHeaders &apos; in LazyRequest # headersSupplier
Modified LoadController.java to use mod.getModId ( ) instead of hardcoded mod id for logging and activating mods .
Added retry status to FileDownloadStatus and updated isKeepAhead and isKeepFlow methods to consider retry status .
Modified ParametersList.java to change the check for -D property names from StringUtil.startsWithConcatenationOf to parameter.startsWith.
Update license from GPL v2 to LGPL v2.1 in Restore.java
Removed unnecessary constructor and logger statement in Options.java
Added test case for cookies parsing supporting no value cookies .
Updated property file path for AmazonKinesisFirehoseToRedshiftSample and AmazonKinesisFirehoseToS3Sample classes .
Renamed theme &apos; Tomorrow Night Eighties &apos; to &apos; Tomorrow Night 80s &apos; in AceThemes.java
Added null check for super type in GenericTypeDescription
Removed unnecessary imports for WebView and LOG in background.java and basicauth.java
Update Groovy version from 1.8.0-rc-1 to 1.8.0-rc-4 in GrailsCoreDependencies.java
Added import statement for OStorage and updated code to use ODistributedStorage instead of OStorageEmbedded
Added support for lowercase .gpx file extensions in MapActivityLayers .
Removed unnecessary documentation and exception handling in JavadocTagInfo and AbstractJavadocCheck .
Modified TagViewFragment to return false for hasDraggableOption when tag data is null or featured
Replace injection of MessageSource with explicit retrieval from application context
Added space character check in OSchemaShared.java
Fixed layout direction for vertical scrollbar
Added loop counter to prevent infinite loop in Encoder.java
Changed layout of JBRunnerTabs to adjust height and width of tabs when hideTabs is false .
Update notification settings to enable notifications by default and only notify about favorites
Increased number of thread executors from 1 to 5 in ApplicationImpl .
Added deleteDirectory method to handle temp directory cleanup .
Added ActionNotFoundTransportException and updated MessageChannelHandler to throw exception instead of logging
Remove unnecessary preloadTexture method
Added parameter &apos; FOONAME &apos; to StringTemplateParser and updated ParamTest to use it .
Added import statement for MessageDefinition and modified createResponseMessage method to use it .
Fixed invalid file path handling in MasterInfo # getClientFileInfo .
Removed unnecessary imports and cleaned up code
Removed protected account check for retweets
Added newline character to printf statement in KafkaConsumer.java
Added logging statements for debugging purposes in AbstractCacheMonitor and AbstractSamlObjectBuilder classes .
Added import statement for android.text.TextUtils ; and fixed a bug where network name was not being updated correctly .
Stopping SpiceManager when activity dies prematurely and unbinding will occur asynchronously .
Replace ColorDrawable with c.drawColor ( ) for setting background color in ThumbnailsCacheManager .
Simplified generateDefaultLayoutParams ( ) method by removing unnecessary vertical / horizontal checks .
Added test for WebSocket version negotiation
Added new method getPercentageCacheMisses and updated getPercentageCacheHits to use it .
Added warning for loading DefaultNodeExtension with different class-loaders .
TupleInfo : Changed return type of slice method from slice to offset + slice
Added LocalQueueStats getter method to DataQueueProxy
Added method getGroups ( String user ) and updated setConf ( TachyonConf conf ) in GroupMappingService interface .
Added support for @ Autowired injection based on ServletContext
Fix FunctionType isStructuralType ( ) to return true for structural interfaces .
Fixed issue with NotificationsPendingDraftsService by removing unnecessary &apos; completed ( ) ; &apos; statement .
Added import statements for ChannelBuffers and Assertions in RedisCodecTest.java
Added &apos; CANNOT _ JOIN &apos; constant and updated &apos; tryJoinRawLines &apos; method in JoinRawLinesHandlerDelegate interface .
Added GitHub token validation to ReleaseNotesBuilderFactory
Removed unnecessary &apos; e.printStackTrace ( ) &apos; statement in ApiHelper.java.
Added import statement for java.util.ArrayList and modified the getAllModules method to return a list of modules instead of a map .
Added recommendation to suspend AtmosphereResource before adding it to a Broadcaster .
Fixed issue with YAxisRenderer calculating first and last values when interval is 0 .
Re-enable internal plugin &apos; lits &apos; for language plugins integration tests .
Renamed module &apos; $ { oldModuleName } &apos; to &apos; $ { module.getName ( ) } &apos; . Also updated the corresponding Eclipse project file .
Added import statements for FacesMessage and ConverterException , and updated the getAsObject method to handle NumberFormatExceptions .
Fixed animation time calculation to account for negative times and added wrapping
Remove unnecessary imports and abstract classes in TransportMessage , TransportRequest , and TransportResponse .
Update AbstractStoreClientFactory to use MetadataStore.STORES _ KEY instead of hardcoded store name for bootstrap metadata fetching
Added --all option to GitHistoryUtils when no branch is specified
Update Registry.java to change copyright year and add support for Java 5 primitive types .
Renamed method &apos; getCompletePredicate &apos; to &apos; getHandledPredicate &apos; in class HandledPredicate .
FMLServerStartingEvent : Changed import from cpw.mods.fml.common.LoaderState to net.minecraft.server.MinecraftServer and added getServer ( ) method returning MinecraftServer instance .
Added import statement for water.Iced in Type.java and changed AST constructor to take a Type parameter instead of a transient Type parameter .
Fixed issue with clipping to transparent views in StackScrollAlgorithm .
Removed unused import statements in JobsHandler.java
Added fencing mechanism to ensure pending invokeFunctor ( ) messages are processed before detaching Functor .
Update CalligraphyContextWrapper to use a different inflater
Added method createStorageDirPath ( String ) to create storage directory path including necessary parent directories with appropriate permissions ( 777 + StickyBit ) .
TransactionHandle : Renamed isPristine ( ) to isImplicit ( )
Fixed null pointer exception in ImageHelper.getThumbnailForWPImageSpan by checking for null context and filePath
Added import statement for JavaLanguage and updated lexer creation for Java files
Added @ Produces ( &quot; application / json &quot; ) annotation to ResourceChat class and updated broadcast method to return a Response object .
Removed inherited role and user settings from RestrictedTest .
Modified AndroidGraphicsGridSampler to use checkAndNudgePoints instead of checkEndpoint
Added jar file to job manager IT case .
Added try-with-resources block to properly close InputStream in ResourceHttpRequestHandler.writeContent ( ) method .
Fixed HTTPServer.java file by removing unused variables and setting runInternal to false by default .
Added import statements for Service and ModelRegistry in JvmComponentPlugin.java and PlayApplicationPlugin.java.
commit ProxyInjector.java modId annotation change
Replace ResolveImportUtil.resolveImportElement with resolve ( ) in PyImportElementImpl .
Removed unnecessary System.out.println statement in ReplayingDecoder .
Added test for setting field value for id using ReflectionUtil
Fixed bug in BreakpointPropertiesPanel : now correctly handles breakpoints with empty conditions .
Fixed license header in BaseIndex.java
Added nullability annotations for AttributeSet parameters in Toolbar constructor .
Added recognition of arguments in GenericTypeOperationHandler and BatchEditLineHandler
Removed unimplemented post method and added missing return type for getHandle method .
Presence.java : updated setPriority method to validate priority value ( -128 through 128 ) and throw IllegalArgumentException for invalid values .
Added logging configuration and updated ConnectionPoolConfig constructor
Modified layout of RecentsView to account for system insets .
Registered SerializerRegistrationsMessage.Registration class .
HttpConnection.java : Handle InterruptedIOException and TimeoutException with proper exception hierarchy
Added setUVRange method to MeshBuilder and MeshPartBuilder
Added ValueAnnotation for semantic graph
AddCallSuperQuickFix : Changed parameter list from superInit.getParameterList ( ) .getParameters ( ) to PyParameterList.getParameters ( ) and added null check for paramList
Updated CommittedChangesTreeBrowser to use VcsBundle and VcsDataKeys
SetType.java : Throw illegal argument exception for out-of-bounds choice values .
Added support for keyword completions in WordCompletionContributor .
Update CppConfiguration.java to correctly handle --sysroot option .
Added method setMessageId ( String ) and updated setInReplyTo ( String ) to use header instead of messageId
Added VisibleForTesting annotation to TupleDomainOrcPredicate.java
Cancel keepAliveTimeout in SearchContext.release ( ) method .
Added registration state check before handling URI for SIP protocol
Remove unnecessary synchronization in OperatorContext.run ( )
Added check for null node before adding or setting item in DocumentOutlineWidget .
Throwing ZException for IOExceptions in ERBEvaluator # eval
Modified NewChunk.java to change the definition of &apos; isConstant &apos; from &apos; sparseLen ( ) = = 0 &apos; to &apos; ! sparse &#124; &#124; sparseLen ( ) = = 0 &apos; .
Modified TachyonFile.java to change parameter name from &apos; blockId &apos; to &apos; blockIndex &apos; in getFilename method .
Added random sample data to SimpleActivity
EditorImpl : Ensure preferred height is at least a single line height
Added setPairedInfo method and updated loadStoredConfig and saveStoredConfig methods in MaxCulBindingConfig class .
Modified AbstractAccessLogValve.java to change the format of the date and time element in the access log from % t { format } to % { format } t .
Removed unused activeCallSessions map to prevent memory leaks .
Added Apache License header to CallableMessageSender.java
Added inspirational link to Java tutorial in QuoteOfTheMomentClient and QuoteOfTheMomentServer classes .
Replace target class loading with member &apos;s declaring class
Added showOrHideQuotedText ( ) function call in MessageCompose.java
Added LogUtil import and modified handling of unknown SharedPreferences changes .
Added check for JavaxScripting availability in Activator.java
Fixed issue with Search Everywhere action in main toolbar
Added support for XML language and parser definitions .
Fixed issue withXmlRootElement annotation , now correctly constructing Xml object and attaching it to the model .
Added debug logging for incoming queries in AsyncQueryForwardingServlet .
Fixed issue with incorrect UID handling in AppWidgetServiceImpl .
Added test suites for ImmutableSet and ImmutableSortedSet .
Removed unused import : android.text.TextUtils
Added method getContextProperty ( String , String ) and updated getContextProperties ( ) to use signature + name for property keys .
Fixed issue with Tree.leftChildIndent not being updated properly in DeferredIconImpl .
Removed unused import and variable , and simplified TransactionCommitment.publishAsClosed ( ) implementation .
Modified DataServerMessage.java to properly close files in case of exception
Removed unnecessary import statement for &apos; edu.stanford.ejalbert.BrowserLauncher&apos;
Fixed issue with HiveZeppelinDriver &apos;s getLocalWarehouse method . Now returns absolute path of warehouse directory .
Fix issue with incorrect assertion in BaseCheckTestSupport.java.
Fix code item identity and remove unnecessary instructions
Replace filename.toString ( ) with file.toString ( ) in AttachmentView.java
Fixed deadlock issue in SquidDatabase.acquireExclusiveLock ( ) by disallowing calls from threads that already hold non-exclusive locks or are in a transaction .
Update AbstractMinMaxAggregation to use generic AccumulatorStateSerializer and AccumulatorStateFactory
UpdateRequest : Added validation for script and doc parameters
Update SHA-1 hash for Maven jar &apos; guava &apos; from &apos; d664fbfc03d2e5ce9cab2a44fb01f1d0bf9dfebeccc1a473b1f9ea31f79f6f99 &apos; to &apos; cce0823396aa693798f8882e64213b1772032b09 &apos; .
Added ROOT _ ACCESS constant for domain admin
Disable roaming for Usage Statistics Persistence Component .
Added startServer ( ) call to GraylogControl in RestAssuredSetupRule before method
Modified Graph.java to clarify that a Graph is never equal to a ValueGraph , and added a check for this in the equals method .
Update copyright year in PopupComponent.java from 2014 to 2015 .
Added &apos; Euch &apos; and &apos; Neuem &apos; to exceptions in CaseRule .
Added JSON mime mapping
Fixed ImagePresenter to properly handle null streamOpeners and missing IDs
Added support for OMEMO encryption by including fingerprints of axolotl service in Account entity .
Added test for recompute executor creating a recompute plan and launching the recompute job at heartbeat .
Removed unnecessary documentation and reformatted code for IOLimiter interface .
Fixed bug in UnionTest.java by removing unnecessary System.out.println statement .
Added generic type parameter to isCompatible and putAll methods in BloomFilter class .
Fixed calculation of turn cost for plain street edges .
Removed unnecessary System.out.println statement in AbstractCoberturaParser .
Renamed import from &apos; com.intellij.testFramework.fixtures.LightPlatformCodeInsightFixtureTestCase &apos; to &apos; com.intellij.testFramework.fixtures.LightCodeInsightFixtureTestCase &apos; and added a new method &apos; setUp &apos; to initialize the test environment .
Added note about API freeze and non-Beta status for Cache and LoadingCache classes .
Added field &apos; currentOperationHandler &apos; to OperationThread for handling operations .
Removed unimplemented method setCarSpeed in TestStreetMatcher .
Update IOStatistics.dump ( ) to include percentage of used leaf nodes .
Removed unnecessary log statement in Chronometer.updateText ( )
Fixed bug in DeleteResponse.java , now returns true if a doc was found to delete instead of no doc found .
Removed unnecessary progress logging statement in DeepLearning class .
Added support for GridLayoutManager and StaggeredGridLayoutManager
Update AppLog to use PackageUtils.getPackageInfo ( context ) instead of PackageManager.getPackageInfo ( context )
Modified ParserATNSimulator.java to return Pair &lt; ATNConfigSet , ATNConfigSet &gt; instead of Pair &lt; List &lt; ATNConfig &gt; , List &lt; ATNConfig &gt; &gt;
Added GdxRuntimeException handling in Gdx2DPixmap constructor .
Added return statement to GELFProcessor to skip incomplete messages
Added DiagnosticGroup for duplicate variables and updated TypeCheckTest to include new tests for duplicate static property declarations and local variable declarations .
Added performance testing for uploading feature .
Wallet.java : Ignore missing connected output in transaction inputs
Removed BuildConfig generation for AAR dependencies .
Fixed issue with null transaction in SingleEntityStoreFromBlocking
Renamed QueueConfig to getQueueConfig and updated the import statement .
Fixing incorrect wording in OConcurrentModificationException message .
Added user principal to RequestContext when upgrading request for WebSocket connections
Update Launcher.java to use spl instead of ZendExtension
Replace usage of Codehaus Plexus StringUtils with Apache Commons Lang StringUtils .
Added blockLength and offset tracking to JavaGenerator .
Removed import of XmlElementDescriptor
Throwable handling improved in ExpressionInterpreter.java
Added import statement for com.github.dreamhead.moco.HttpMethod and updated method signature to use com.github.dreamhead.moco.HttpMethod instead of io.netty.handler.codec.http.HttpMethod
Added handling of OutOfMemoryError in MimeUtility.getTextFromPart
Modified GDataImporter to use getTableRow instead of getCellValue
Added method to append values to path-like environment variables .
Update FileDownloadTask to use the last task in the list for update instead of the first one .
Modified LogAnalyzerTotal.java to use new Functions.IpContentTuple ( ) instead of Functions.IpTuple ( ) for joining transfer amount and request count by IP address .
Added default node and edge label associations for yEd from yworks.com.
Added Proxy support to Connection interface .
Added null check for row elements in OtpsCsvOutput.addRow ( ) method .
Added extension point for base panel and updated registerValidators method .
MBeanDumper : Changed error handling for getting attribute to use log.error ( Throwable t ) instead of log.error ( String message , Throwable t )
Added file creation and directory creation logic to Reporter class .
Added maxTransfers field and setter / getter methods to Request class .
Added constants for unsigned byte , short , and int max values .
Fixed scaling issues in View , WebViewClient , and MediaPlayer .
Added LABEL family to Features class .
Added attrib ( ) methods for setting shape attributes .
Added support for serializing default states and boolean values in XML .
Lowered default size of IndexedRingBuffer from 256 to 128 for Android ( https : / / github.com / ReactiveX / RxJava / issues / 1820 )
Replace nodeidx = 0 with nodeidx = i in GetLogsFromNode constructor
Remove deprecated method cacheIsNotNull ( ) from PyPackageManager and PyPackageManagerImpl .
Added import statement for Factory and removed unused import statement for InvocationTargetException . Also , modified the doInvalidate method to use the DomManagerImpl.getDomInvocationHandler ( myCachedValue ) instead of directly using the DomElement.detach ( true )
Increased maxIdleTime from 1 second to 200 seconds in HttpConfiguration .
Added check for empty GCM token before registering device token on remote services
Added profile loading logic to SimpleHeaderDrawerActivity .
Added support for concatenating null literals in strings .
Update StorageService to call SystemTable.updateToken in server mode
Added ModelLoader.White.instance.register ( map ) to ForgeHooksClient.onTextureStitchedPre and removed MinecraftForge.EVENT _ BUS.register ( White.instance ) from ModelLoader constructor .
Fixed debug logging for RabbitMQEndpoint creation
Added filter to Palette class for fine-grained color control .
Renamed &apos; toast &apos; field from &apos; private static final Toast toast &apos; to &apos; private static Toast toast &apos; in Toaster.java.
Added support for floating-point increments with INCRBYFLOAT command .
Added import statement for com.google.common.collect.Maps ; and replaced ImmutableMap.Builder with Maps.uniqueIndex for creating properties map .
Added &apos; setUser ( true ) &apos; to TaskstJob constructor
Added explore.web.browser and com.jiubang.browser to Util.java
Fixed error message in ItemUIRegistryImpl to include the item name in the log message .
Fixed indexing stamp cache flush in FileBasedIndex
Added @ SuppressWarnings annotation to MongoDbClient.read ( ) method
Remove unnecessary import and release of keyBuf in BinaryMemcacheEncoder
Simplified conditional logic in Comment.java by removing multiplication by Integer.MAX _ VALUE .
Added support for handling XML attribute values in TypedHandler .
Added support for multiple blogs in Helpshift metadata
Added new settings for checkin : CHECKIN _ SEND _ LAST _ KMSG _ TIME , CHECKIN _ SEND _ APANIC _ CONSOLE _ TIME , and CHECKIN _ SEND _ APANIC _ THREAD _ TIME .
Added a fix to prevent exceptions when user types ! foo in the shell command .
Added profiler config to MockTraceContext
Fix media type creation to allow wildcard subtypes with non-wildcard types
Modified MembershipResource.java to remove unnecessary HTTP response annotations .
Added persistent UI for scrollbars in EditorImpl and EditorMarkupModelImpl .
Modified hashCode ( ) method in DFAState class to use configs.hashCode ( ) instead of iterating over configs array .
Added soft keyboard opening in MainActivity &apos;s onClickClear method
Fix for issue # 12345 : Optimized closure execution in ConfigureUtil by rehydrating closures with a new owner .
Added field &apos; myLoaded &apos; to ModuleRootManagerImpl and updated loadState method to use it .
Added documentation for subclasses to throw a subclass of RuntimeException to allow creating a CheckedFuture that throws both checked and unchecked exceptions .
Added check for deviceinfo ! = null before setting zone count and updated logger message
Added support for synthesizing method parameters in Spring Core .
Renamed variable &apos; to &apos; to &apos; destPointsetId &apos; in OneToManyProfileRequest constructor .
Fix null pointer checks for ReilInstruction constructors .
Added import statement for &apos; TesterServletContext &apos; and updated &apos; DigestAuthenticator &apos; to use &apos; TesterContext &apos; instead of &apos; Container &apos; .
Added field mProperSigningKeySet to class PackageKeySetData and copied its value from original object in constructor .
Replace hardcoded logger name with dynamic name based on servlet name
Added getter for HttpClient in HttpClientTransportOverHTTP2
PCA does not work on categorical data
Added import statement for IdeaTestFixtureFactoryImpl and removed unnecessary code
Allow SchemaManager to return true if plugin doesn &apos;t need database tables or if the schema is already up-to-date .
Modified RemoteBlockInStream to use mCheckpointPos as the position in the checkpointed file relative to the block , and added a skip call to move to the beginning of the block in the checkpoint file .
Added WPPrefUtils.layoutAsNumberPickerSelected ( ) call to WPNumberPicker constructor and setValue ( ) method .
Change total bytes to capacity bytes in WorkerInfo and fix memory unit conversion in CommonUtils .
Fixed LdapResource deleteLdapSettings method to properly clean up LdapAuthenticator settings .
Fixed issue with sticky sessions in WebFilter .
Modified SamlIdPProperties to use TimeUnit.DAYS.toMinutes ( 1 ) for cacheExpirationMinutes instead of a hardcoded value of 30 minutes .
Set expand events to false in ReplFromTerminal
Fixed missing or invalid timestamps in Module.java and RepoParser.java
Re-write FIXME comments for BikeFlagEncoder class
Removed unnecessary import statement for &apos; org.springframework.util.Assert &apos; in ServiceTicketImpl.java.
Added import statement for PatternSyntaxException and updated code to handle errors in POS tag synthesis .
Added column name and min / max values to summary page
Replace backToLocationImpl with getMapLayers ( ) .selectMapLayer ( view ) in MapInfoLayer
Added import for Serializable and implemented Serializable interface in RenamingMap.java
Added movement method to tvInfoMethod and set visibility to VISIBLE if expert
Added import statement for OkUrlFactory and replaced HttpURLConnection with OkUrlFactory .
Added check for longest chunk in RelationTriple.java
Added support for non-physical elements in PsiAnchor.createStubReference ( ) method .
Added suffix .bpmn to BpmnDeployer.java for better compatibility with BPMN world ( Eclipse Designer , Signavio , jBPM 5 )
Added inflate and invalidate methods for updating downloads list in DashUpdatesFragment .
Added &apos; hide &apos; annotation to WebHistoryItem constructor
Added OnPostPopupListener to ReaderPostAdapter
Added persistId field to Inputs.java
Fix issue with empty cluster detection in FuzzyKMeansMapper
Fixed bug in ChunkedStream.java : incorrect offset calculation .
Fixed navigation bar behavior in NotificationsDetailActivity .
Modified TemplateDataElementType.java to append a newline instead of a space when appending template tokens .
Removed unused import org.omg.CORBA.MARSHAL and updated import com.ea.orbit.actors.annotation.IdStrategy to match the actual class name .
Added new fields for allDay events in Calendar class
Added import statement for &apos; checkArgument &apos; from &apos; com.google.common.base &apos; and updated &apos; getVariable &apos; method to use &apos; checkArgument &apos; instead of &apos; Preconditions.checkArgument &apos; .
Added import of ClosureCommand and updated receiver to use ClosureCommand in RemoteControlHandler .
Added visibility check before drawing frame in ImageWallpaper .
Added import for com.thinkaurelius.titan.core.attribute
Modified CopyStep.java to use logError instead of e.printStackTrace and added a new argument to getDescription to provide a human-readable description of the step
Added PATH _ TO _ WEB _ INF _ VIEWS to search paths in DefaultGroovyPageLocator
Added placeholder for animating PicassoDrawable
Added write access assertion for VFS events in PersistentFSImpl .
Added &apos; IGNORE &apos; clause to INSERT statements in SeriesGuideDatabase class to avoid duplicate entries in EPISODES _ SEARCH table .
Disable PyUnboundLocalVariableInspection by default
Added test for main ( ) function with 3 lines
Optimistic transaction : bypass local buffer for commits
Changed log file name from &apos; logBaseName + logNameSuffix &apos; to &apos; logBaseName + this.logNameSuffix &apos; in DefaultAccessLogReceiver constructor .
Fixed byte shift in ByteArrayInput.java
Added import statement for CaseInsensitiveKeyMap and replaced Map &lt; String , List &lt; String &gt; &gt; headers with CaseInsensitiveKeyMap &lt; String , List &lt; String &gt; &gt; in ApplicationPushBuilder .
Added code to getServerName ( ) method to handle cases where the host header contains a port number .
Added MathUtils.PI2 and updated it to be a static final variable .
Removed setting for node client .
Removed unused import statement for DefaultItemAnimator and updated RecyclerView &apos;s ItemAnimator to use the default animator .
Modified RT.java to load classes using Class.forName ( ) with false flag set for bootstrap class loading
Modified ServerSystemTest to use &apos; Compute Engine Database Connection &apos; instead of &apos; Compute Engine Database &apos; in response content check .
Maps : Changed internal backing from EnumSet to EnumMap .
Added line titledSeparator.setText ( title ) in IdeaTitledBorder.java
Removed unnecessary import statement for &apos; java.io.File &apos; in ResourceModelEncoder.java.
Added visitLines method call in NoSonarFilterLoader to handle suppress warnings
Added return statement for thr.mService in WindowManagerService.java
Added port collision check after checking for job deployment conflict
Removed unused attribute &apos; stat _ playlist &apos; from NavListAdapter.java
Added support for implicit references to tag files in JspFile elements .
Added check for non-null scheduler before registering task manager
Fixed issue with focus loss animation not starting from current value
Added method commitAllDocuments ( ) to LightIdeaTestCase and cleared caches in InjectedLanguageUtil .
Added predefined lexer rule properties dict to Grammar and Rule classes .
Added selectKeys method to TableStatements class
Added &apos; continue &apos; statement in catch block to skip over unknown elements during configuration settings loading .
Updated LocalSession class with new expiration time calculation and added CUI support .
Added scaled image and scaled image img width columns to WordPressDB .
Added check to ensure UdpConnector only receives packets from remote address
Added getClassSize ( ) method to DexFileReader and updated AsmVerify to use the new method .
Update comparator for pathsOrderedByLength in ControllerRestApplication
100644
Fixed bug in BusinessRuleTaskActivityBehavior.java by setting accept parameter correctly .
Added link to StoreView.html for H2O.SELF in Cloud.java
Fixed equality check for DeferredType
Removed unused import statement for GraphDatabaseSettings
Increased the default polling period for waiting for file completion in TachyonFileSystemUtils from 1 second to a generous 1 second .
Fixed assertion in DefaultServiceRegistry.groovy
Removed unused import statement for &apos; com.hazelcast.nio.Packet &apos; in ReadHandler.java.
Added support for dropping plain text and HTML text into the editor
Restarting password null check .
Fixed issue with interrupted mid-tag in XmppConnection.java
Disable TestUserApiAddOn tests for now , will re-enable after merge
Added ability to pass boolean parameters to fieldStats ( ) method in FieldValueAlertConditionTest .
Added Systrace and HMRClient to CoreModulesPackage
Added rect ( ) and ellipseMode ( ) methods to PApplet class .
Fixing error handling in MethodAnnotationAggregator by throwing a RuntimeException instead of logging an error .
Removed Ajp13Connector and CoyoteConnector from MBeanUtils.java exceptions array .
Fixed error message in ScriptImpl.java to use a more descriptive message for script execution errors .
Added propagation of original attributes to WebSocket message in SimpleHttpProtocol .
Added import statement for org.apache.camel.Endpoint
Added com.esotericsoftware.reflectasm.MethodAccess class to moduleSecurityManager
Renamed method from &apos; fizzbuzz &apos; to &apos; fizzBuzz &apos; and updated import statements .
Fixes # 12345 : Remove unnecessary reentrant lock in Daemon.start ( ) method .
Remove unused variable &apos; taskStatusNagger &apos; from TaskInternalServiceRegistry .
Update MLUpdate.java to push PMML model onto update topic only if it exists and is not too large .
Fix for DNS name resolution : Instead of returning a substring of the original name , we now return the entire name as a string .
Added import statement for AstridPreferences and updated showListSettingsPopover method to check if popover can be shown before displaying it .
Activity.java : Restored state initialization from onStart to onCreate .
Remove deprecated method onViewCreated ( View ) in HeaderTransformer and use onViewCreated ( Activity , View ) instead .
Added import of com.android.okhttp.OkResponseCache ; in HttpResponseCache.java
Added LuceneTestCase.AwaitsFix annotation to IndicesLifecycleListenerTests to await fix for bug URL &apos; fails due to old ping responses confusing master elections , bleskes investigating&apos;
Audit event context map updated to include principal name instead of principal object .
Added field &apos; endReached &apos; to class TableInputFormat to keep track of whether the end of the table has been reached .
Added sorting by name in ConstantPoolTest
Added import statement for &apos; javax.annotation.ParametersAreNonnullByDefault &apos; in package-info.java.
Added test case for fake failure in StringToolsTest
Renamed &apos; isStarted &apos; field from atomic boolean to final atomic boolean in AbstractService
Added import statement for AuthToken and updated SessionManager to consider sessions that never authenticated as anonymous .
Added DummyBlockReplacementTest to RegistryTestSuite
ChangesViewManager : Save splitter proportion on project closure
Added buffer size parameter to NativeGCMCipherHelper getOutputStream method .
Updated list of map names and languages
Added support for double constants in StackManipulation and updated ByteBuddyTest to use DoubleConstant .
Added getContextPath ( ) method to JspCServletContext class .
Fix bug in EncodedDataType.java : incorrect calculation of encoding size
Added support for annotating candidate action methods in ControllerActionTransformer .
Added &apos; newerThan &apos; parameter to YammerMessagesConsumerOptionTest
Added public access modifier to method incomingEdgeList ( IndexedWord v ) in class SemanticGraph .
Added parameter descriptions to TargetMethodAnnotationDrivenBinder constructor .
Added method &apos; equals &apos; to class PrimitiveValue and added a new method &apos; toString &apos; with throws IllegalArgumentException . Also , updated the existing &apos; toString &apos; method to include a check for unknown representation .
Change scanner table from uidTable ( ) to metaTable ( )
Added cache for daemon registries
Replace background resource with button drawable in TaskAdapter and EditTitleControlSet .
Added support for transaction isolation levels in ConnectionPageGeneral .
Modified FileUtils.java to use paths instead of files .
Updated regular expression for splitting words with special characters .
Renamed class ReactiveDialog to RxDialogFragment and updated import statements .
Added null check for getStructFieldData and getStructFieldsDataAsList methods in OrcStruct class .
Removed unnecessary hooks for clipboard manager .
Fixed issue with connection pool in JedisSlotBasedConnectionHandler
Fixed issue with left click air swings not being cancelled properly .
Remove unused parameter in UberRDotJava
Added methods getSuccessHandler ( ) and getFailureHandler ( ) to AbstractAuthenticationProcessingFilter .
Remove LoadBroadcaster.instance.getLoadInfo ( ) from BootStrapper.getBootstrapTokens ( ) method
Added import of java.security.AccessControlContext ; and updated checkPermission method to use engineContext.checkPermission instead of sm.checkPermission
Added empty onStartTrackingTouch and onStopTrackingTouch methods to FillableLoaderPage
Added error logging for creating account
Merge pull request # 123 from branch &apos; feature / new-file-system &apos; into &apos; master &apos; . \ n \ nThis commit includes changes to the TachyonFileSystem class , specifically in the createEmptyFile and getOutStream methods . The changes include removing unnecessary code and simplifying the implementation . Additionally , the createEmptyFile method now calls the getOutStream method instead of creating a new FileOutStream instance .
Added registration id to ClientService.addClientListener method .
Added NativeServices.NATIVE _ DIR _ OVERRIDE to JvmOptions &apos; IMMUTABLE _ SYSTEM _ PROPERTIES
Added function loading mechanism
Optimize match all query for the &apos; * : * &apos; and &apos; * &apos; cases .
Added maximum source pane feature for R Markdown notebooks
Added alpha value setting for currDrawable in DrawableContainer
Added logic to show self-hosted sign-in form for username-email login .
Implement getKey ( ) method in GeneratedMethodBinding class .
Added test for invalid surrogate pairs in RealmObjectTests .
Added static variable &apos; EMPTY _ SOURCE &apos; to CommonToken class and updated constructor to use it .
Update NoopTask.java , updated default run time and is ready time , added create method
Handle case where bluetooth service proxy is already null .
Removed unnecessary destroy ( ) method from FilterBase .
Update IS _ WINDOWS variable to use System.getProperty ( &quot; os.name &quot; , &quot; &quot; ) instead of System.getProperty ( &quot; os.name &quot; ) to avoid issues on Windows .
Renamed &apos; setCamera &apos; and &apos; getCamera &apos; methods to &apos; setLight &apos; and &apos; getLight &apos; in LightNode class .
JOGL 2.0 : Remove null-terminated strings in glGetAttribLocation and glGetUniformLocation
JarFileSystemImpl : Extract root path correctly for case-insensitive paths
Update copyright year in FormatProcessor.java from 2012 to 2015 .
Renamed &apos; setTextColor &apos; method to &apos; setTextColorResourceId &apos; and updated parameter name accordingly .
Added getDisplaySize ( ) method to UIDisplayElement , returning the display size in pixels .
Removed unused import statement for &apos; org.apache.tomcat.util.buf.CharChunk &apos; in FormAuthModule.java.
Added NUMCAST and STRING _ CAST operators to Ops class .
Added static modifier to private inner classes EmptyStateInitializer and CopyStateInitializer in PartitionStateGeneratorImpl.java
Added license header and updated package name in SSLContextParametersSecureProtocolSocketFactory.java
Renamed enum classes in Searches.java from static final to public .
Fixed incorrect lifecycle handling in WebSocketProcessorTest .
Added debug statements for DNS SRV resolution and removed unnecessary code .
Ensure that BOSH communication does not have Chunked encoding by setting the buffer size of the response to the length of the data .
Added check to only add palettes with at least PALETTE _ DISPLAY _ LIMIT colors to the recent palette list
Increased execution time limit from 500 to 1000 and updated port number to 7476
Fixed brace matching for empty elements and CDATA sections .
Reverse order of siblings in ResourceBundlePropertiesInsertManagerImpl
Modified KubernetesEndpoint.java to use &apos; kubernetes : masterUrl &apos; instead of &apos; kubernetes : master &apos; in the UriEndpoint annotation .
Fix for declaration resolution when package is defined out of Kotlin sources
Added method to refresh notes and registered receiver for bucket change events in NotificationsListFragment .
Fixed issue with L-BFGS solver and multinomial family in GLM model
Added fast / dom / DeviceOrientation / basic-operation.html to ignore list
Fix bug in NettyBuffer.readBytes ( ) method to properly retain reference to buffer .
Removed unnecessary import statement for &apos; TextConsoleBuilderFactory &apos; in JetRunConfiguration.java.
Fixed issue with marking root directory dirty in FileWatcher .
Added android project files to GdxSetup .
Removed unused imports : Pref , SomePrefs _
Added support for transport networks and updated GUI handling .
Added JUnit4 runner and updated import statements for ConcurrencyTest .
Fixed issue with isWritable ( ) method in LightPsiFileImpl.java
Replace PrintStream with ByteArrayOutputStream in MainRunner .
Removed import of &apos; org.mongodb.operation.Find &apos; in PinnedSessionTest.java
Update DefaultTransform.java to calculate transformation correctly when scaling down .
Added annotation processing for ClassInjector
Added support for forward distribution strategy in OutputEmitter .
Added Apache License header to AsyncImageView.java
Modified DKV.java to change the behavior of the remove method to return null instead of a Value object when put ( key , null ) is called .
Added @ Ignore annotation to BlueprintProduceConsumeImplicitPropertyTest # testImplicit method as it fails . Fix Me ! ( diff from 4d292615fb to e90450ebd9 ) .
Added reflection to call close ( ) method on SocketChannelImpl $ SocketAdapter
Modified QuickStatusBarHeader.java to use mExpandIndicator instead of this as the header view
Added test for issue https : / / github.com / grpc / grpc-java / issues / 615 : completeWithoutClose
Added support for externalizable HyperLogLog objects
Added support for multiple containers in DesktopGeneratorController .
Added frequently ( ) method call to setMergePolicy for FreqTermsEnumTests
Repair of graph completed in { time } secs \ n scannedEdges ..... : { scannedEdges } \ n removedEdges ..... : { removedEdges }
Added logging statements for generated random tokens and updated format of existing logging statement .
Added compatibility redirect for executionTimeoutInMilliseconds ( ) in HystrixCommandProperties .
Added support for soft input mode in EditingPOIActivity .
Added error handling for missing session during registration
Added import statement for javax.annotation.CheckReturnValue in Defaults.java
Added timeout parameter to HttpConnection.shutdown ( ) and changed close ( ) method to close all active streams before sending GOAWAY frame .
Added debug logging for PassExecutorService in testCodeFoldingPassRestartsOnRegionUnfolding
Added check for break char in BitmapFont.isWhitespace ( ) and added a new method isBreakChar ( char ) to return true if the character is a break char .
Fix empty tree list
Assertion added to test toStringMatches _ withDeadline in CallOptionsTest.java
Added hasGravity ( ) method to Material enum .
Added check for null tries before adding tries to method definition .
Rename &apos; notFoundMarker &apos; to &apos; NOT _ FOUND _ MARKER &apos; in SPVBlockStore.java
Added project parameter to BinaryFileStubBuilder.buildStubTree method .
Added error handling for look and feel exceptions in Main class .
Added setupStructurePopup method to FileStructureTestBase class .
Fix buffered description handling in Utils.java
Added permission for WorldGeneratorPluginLibrary to access declared members .
Removed unnecessary System.out.println statement in AggregationStrategyWithPreservationTest .
OF-898 : Replace any number of millisecond-characters with at most three of them .
Added test for single SSL socket factory
Increased connectionsPerThread from 1 to 16 in LoadBalancingProxyAJPTestCase .
Remove unnecessary tap callback in cancelLongPress
Added function to determine whether n / 2 distinctive pairs can be formed from given n integers where n is even and each pair &apos;s sum is divisible by given k . Also , added functionality to convert base 9 numbers to base 10 , taking care to subtract 1 when multiplying numbers from 5 to 9 .
Added @ PreDestroy annotation to EmbeddedDatabase interface and implemented shutdown method .
Fixed deserialize method in StringMultiSchemeWithTopic to throw UnsupportedOperationException instead of NotImplementedException .
Removed unused import statement for AcceptEncodingGZIPInterceptor
Removed unused import and variable , and removed unnecessary onConfigurationChanged method
Added getMaxStackSize and setMaxStackSize methods to Inventory interface
Added support for simple intermediate adjective conjuncts
Added operation counter for estimating memory size in MemoryTranslog
Added check for statusBeforeSeeking and set media position accordingly in LocalPSMP # seekTo
Updated VCardEntry to use Im instead of Phone for TYPE _ HOME and added a new field &apos; data &apos; to the Im protocol .
Added support for masking payload in WsRemoteEndpointImplBase
Fix for bug in RelationTriple.java : incorrect index calculation for longest chunk
Added import statement for EntityPlayerMP and updated tickStart and tickEnd methods to include client-side handling .
Modified BasicConfiguration.java to use trim ( ) method before parsing boolean values .
Added ordinal to UDPRebooted.java
Added support for tab and newline characters in string literals .
Removed unnecessary imports : java.util.ArrayList , java.util.Collection , and java.util.Collections
Added range-to-endpoint mapping printout in NodeProbe
Fixed null pointer exception in OBJLoader by adding a check for null objectName and setting a default name &apos; Model &apos; for the node when objectName is null .
LockHolderImpl.java : Changed equals method to use actual instance instead of SchemaLock .
Added placeholder image support in MediaGridAdapter .
Increased maximum event length from 2048 to 4096 bytes .
Fixing file path for EphemeralFileSystemAbstraction
Allow focusing on directory name text field when resource type combo is disabled
Renamed NATS constants and updated references in NatsConsumer
Added null check for otr _ fingerprints in Contact.java
Fixed issue with non-idempotent commands executed via HTTP GET . Changed OServerCommandPostCommand to use db.command ( cmd ) .execute ( ) instead of executor.execute ( null ) for better performance .
Update exception type from AmazonServiceException to AmazonClientException in SimpleDBRecorder .
Disposed of children in LightStubBuilder and LightTreeUtil
Replace UriRequest # &lt; init &gt; with HttpTask # createNewRequest in RequestParams
Added check for isAdded ( ) before setting search results and empty message in TraktAddFragment and TvdbAddFragment
Added check for reminder enabled preference before injecting dependencies in ReengagementReceiver .
Add license header to DynSqlMapper.java
Added import for &apos; com.intellij.util.containers.ContainerUtil &apos; in PackageViewDescriptorImpl.java
Update BaseUpdateRequest.java and FindAndModify.java files .
Added import of ClassUtils and updated ObjectIdentityImpl to use ClassUtils.getUserClass instead of object.getClass ( ) for getting the Java type .
Added import of ActivityTracker and updated InstalledPackagesPanel to use it for emulating activity
Added stacktrace message to GrailsConsole errors
Fix issue with OperatorTakeUntil ignoring other exceptions
Added support for MongoDB 2.8 + server discovery in ListCollectionsOperation .
Added method requestRectangleOnScreen to WebTextView to prevent scrolling during zoom animation and updated WebView &apos;s inAnimateZoom method .
Implemented indexOfAsync method in RedissonList class .
Removed unused import statement for FileViewEnvironment
Update Pager.java to include &apos; Commits &apos; in the page title when displaying a range of commits .
Upgrade Freemarker version from 2.3.21 to the latest version .
Added parameter &apos; enabled &apos; to Status constructor and updated documentation to reflect new functionality .
Added null check for image bytes before creating an image from them .
Added import statement for &apos; java.util.concurrent.RejectedExecutionException &apos; and updated &apos; execute &apos; method to throw this exception instead of logging a warning .
Added import statements for PyClass , PyElement , and PyFunction , and updated the conditions for adding variants from the index .
Added check for JingleCandidate.isOurs ( ) before adding candidate to list of elements
ZoomableDraweeView : Changed onTouchEvent return value from true to false
Disabled transactions in DBMaker.java
Renamed &apos; SwipeFlingAdapterView &apos; to &apos; SwipeAdapterView &apos; and removed unnecessary constructor .
Added method matchingDegree to NameUtil class .
Modified AccountManager.java to return AccountManagerFuture instead of AccountManagerFeature
Removed unused import and method , cleaned up code .
Added server-side SETTINGS frame in FlowControlTest and fixed incorrect handling of Reply frames in HTTP2Session .
Added flag for privileged permissions and removed unnecessary flag
Added a fix for rare disastrous classloading in first call to LockSupport.park.
Fixed GroupMapper to properly handle null values in metadata
Fix for malformed UTF-8 decoding in Utf8Decoder
Updated BuildCraftBuilders.java to change the quarry &apos;s behavior when it comes to chunk loading .
Removed unnecessary call to EasyTracker.getInstance ( ) .activityStart ( this ) ; isesActivity.java
Renamed Pact to Stratosphere in FileOutputFormat.java
Added ComponentListener to PdeBase class for resizing events
Added method getLoadBalancerExecutor ( ) to LoadBalancingRxClient
Removed unused method &apos; disconnect &apos; and simplified &apos; stopAcceptingConnections &apos; .
Fixed empty suite issue in SMTRunnerNotificationsHandler
Added anonymous namespace naming for modules and removed unnecessary code
Added CreateFileOptions list initialization with diamond operator .
Added shouldOverrideUrlLoading method to SmsSync class .
Update ProducerZooKeeper to set new brokers info and close removed brokers .
Added rebalancing redirect for non-metadata stores
Added logging statement in loadMetadata method of BaseFileSystem class
Added exception handling to onDecorateBitmap method in BaseBitmapTextureAtlasSourceDecorator.java
Added option to rewrite labels in smali files from old format to new format .
Added @ Before override for AllocatorContractTest to include strategies initialization .
Added Serializable interface implementation to Person and Address entities .
Restore local changes from stash or shelf
Added link to documentation for using ViewPager with ActionBar tabs .
Added method &apos; withIcon &apos; to ProfileDrawerItem class , allowing for the specification of an icon resource or bitmap .
Added import statements for Arrays , DocumentBuilder , and DocumentBuilderFactory in DirectoryListingTest.java.
Fixed issue with incorrect exception handling in Commands class .
Added support for dynamic app widgets based on Stack Overflow answer
Added Java regular expression import and updated ByNameArrangementEntryMatcher class to use compiled pattern for improved performance .
RefreshableTask : Converted executor to a daemon single-threaded executor .
Remove constant MILLISECOND _ PER _ SECOND and private constructor
Fix HttpPageBufferClient to handle sequenceId after end of response
Added PREV _ APPLICATION _ MODE setting and updated routing logic in MapActivityActions .
Moved RIL request codes for set radio mode , data idle , set UICC subscription , allow data , get hardware configuration , and ICC SIM authentication to new values .
Added SubnetUtils initialization with default value for cases where netmask configuration is not found .
Removed unnecessary import statement for SpreadWildcard .
Added default bounds for virtual views and ensured clients set parent bounds
Fixed issue with color indexing in MapMarkersHelper
Added ConcurrentHashMap for online indexes in IndexStatisticsTest
Removed unused method override for GitRefManagerTest .
Implement semantical check for function calls and validate function calls on objects
Added support for Chromium WebView
Added Ignore annotation to flaky test method testGracefulShutdown .
Added support for resolving flexible types in ForceResolveUtil .
Added immutable field &apos; blah &apos; to enum class &apos; MyEnum1 &apos; and updated VisibilityModifierCheck to allow immutable fields in final classes or enum classes .
Added applicationInfo to ActivityInfo and ShadowResolveInfo
Added optional parameter &apos; incrementMaxConnections &apos; to PeerGroup.connectTo ( InetSocketAddress ) and added a new protected method connectTo ( InetSocketAddress , boolean ) for internal use .
Fixed exception handling in GitSimpleHandler.java
Fixed hint fields to be modifiable
Increased tile size from 15 to 16 in CykloatlasSourceTemplate constructor
Removed unnecessary import statement for &apos; Collections &apos; in JavaCommand.java.
Added suspend ( ) method call to resetAnimator ( ) method in ButtonlessScrollBarUI class .
Added volatile-write for running flag in ManagementCenterService
Modified AbstractDelegatingCasView to use &apos; render view &apos; instead of &apos; model ... &apos; in logger statement .
Added setup and clean methods to OrientGraphJdbcQueryTest
Fix for android bug in ExtraHandler : getIntent ( ) .getExtras ( ) .getByteArray ( ) always returns null ; also added handleByteArrayExtraBug method to handle this issue .
Added import statement for Collections ; updated DomChildDescriptionImpl to use Collections.emptyList ( ) instead of creating an empty list manually .
Fix issue with creating output directory for screen recording
Added synchronized method to PythonNamesValidator
Remove unnecessary casts in TimeRange.java
Fixed issue caching for SonarQube plugins .
Added support for AppCompatActivity and updated layout
Added logic to handle supplicant state change events and updated driver starting procedure .
Increased TAP _ TIMEOUT _ MSEC from 500 to 1000
Added logic to skip over empty lines while reading input file .
Added a try-catch block to handle exceptions when testing connectivity to hosts with multiple addresses .
Removed unnecessary sessionExitpoint call in FalsingManager
Replace AbstractRanking with DynamicAttributeRanking in AttributeRankingBuilder
Added websocketResource method to Utils class
Added support for handling ProcessingExceptions in OrbitRestClient .
Renamed methods from &apos; getKind &apos; and &apos; setKind &apos; to &apos; getToken &apos; and &apos; setToken &apos; in Node.java
Added RequestScoped and SessionScoped annotations to MyRequestScopedBean and MySessionScopedBean respectively .
Hide unused parameters for unsupervised models .
Fixed type mapping for VARCHAR columns
Added logcat command with time stamp and radio logs option .
Added import for org.junit.ComparisonFailure and overridden checkTestOutput ( ) method to handle comparison failures .
Register resource roots for h2o-web and h2o-core training frames
Fix for client request handling : remove unnecessary null checks and improve error handling
Added support for parsing TLDs with taglib URIs in Undertow deployment processor .
Added check for active term ID in determineIndexByTermId method .
Removed unnecessary lock in PaymentGatewayProcessor
Remove unnecessary code in SwipeLayout.java
Renamed TickEvent.Type from static enum to public enum .
Added test for move method and updated dependencies .
Added import statement for GrantedAuthority and updated RoleVoter to use it .
Added isClosed ( ) method to AsyncBatchCursor and AsyncQueryBatchCursor
Added hint to avoid clipping notification backgrounds
Added test for multi-insert with continue
Added error message for too many push jobs happening at the same time in AdminServiceRequestHandler .
Added roster versioning to XmppConnection.java
Added support for sticky broadcasts in ShadowApplication
Added logger and updated OpenCellID markers
Added support for loading lib directory as a Java property
Added support for timeouts in BlockJUnit4ClassRunner
Fix invalid user data exception message
JoglGraphics.java : Changed minor version check for JOGL GL10 creation
Updated SchemaManager to use &apos; jiveVersion &apos; table instead of &apos; ofVersion &apos; for version updates .
Renamed &apos; artifactResults &apos; field from private to public in DependencyResolvingClasspath
Added logging statement for timing test disablement
Added step counter and updated test methods to use it .
Fixed issue with JPQLSerializer not correctly handling EntityPath roots .
Fixed bug in AbstractHttp11Processor by swapping boolean arguments in registerForEvent ( ) method .
Added connectivity check for carrier text
DeclareVariable.java : Removed debug check and simplified return type .
Added flash scope support to FlowScope interface .
Added createdDate and updatedDate fields to BusinessInvoicePaymentDao , with a TODO comment for Pierre .
Fix MuninnPage.java to zero-fill the rest of the page instead of just up to the cache page size .
Added support for Message-ID header in LocalStore
Added UNSAFE.loadFence ( ) to prevent older loads being moved ahead of validate in Receiver.validate ( ) and Receiver.validate ( long cursor )
Remove unused import statement for AzureUnicastHostsProvider .
Added check for null before showing media controller in full screen
Removed unused imports RecursionGuard and RecursionManager in VcsProjectLog.java
Modified SyncManager to check background data usage setting before synchronizing .
Added ability to construct RequestParams with multiple initial key / value string pairs .
Added System.getProperties ( ) to EventTrackerHandler &apos;s properties load .
Added test for multiple underscores in domain name and fixed equality test .
Renamed TransportShardFlushAction and TransportShardRefreshAction to use action name conventions
Added extra fields for recording duration limit and content resolver Uri in MediaStore class .
Update WXSDK version from 0.5.2.2 to 0.5.2.8 in WXEnvironment.java
Removed broken inspection tool factories from InspectionToolRegistrar .
Added TransactionalTask interface for transactional logic execution
Added support for hiding action items related to the content view when the nav drawer is open
Added DiskThresholdDecider configuration to CrateTestCluster
Renamed fields from &apos; private &apos; to &apos; protected &apos; in class KdTree.XYZPoint
Added check for whole word in FindManagerImpl
Added prototype inheritance example using Character instantiation and updated AppTest class .
Added shutdown method to taskExecutor bean and updated SchedulingConfigurer documentation .
Update ContextualUndoAdapter to use the correct content view
Removed unnecessary Log statement in SsoWebViewClient .
Disable on-disc stream cache for faster texture pack reloading .
Added aspect ratio logic for indeterminate drawables in ProgressBar
Update OpenSSLSocketImpl class name for Android platforms .
Optimize subsampling layer by using putiColumnVector instead of addiRowVector
Increased state buffer length and updated offsets in LogBufferDescriptor
Added functionality to write ZWave node data to disk .
Fixed error messages for parameter types in LambdaExpressionCompatibilityConstraint and PsiMethodReferenceCompatibilityConstraint .
Replace \ \ t with \ \ W +
Update TestGremlin.java to change Gremlin script from &apos; g.v ( 2 ) .outE.inV &apos; to &apos; g.v ( 4 ) .outE.inV &apos; in testGremlinPostJSON method .
Modified UIServiceImpl to handle null checks for plugin components and corrected container constraint handling .
Added public access modifier to ENUMERATE _ NESTED _ FILES _ FLAG constant in FileInputFormat.java
Fixed typo in CDFitness.java : changed &apos; fp &apos; to &apos; tp &apos; and &apos; fn &apos; to &apos; fn &apos; in method equals .
Updated Activity.java to use getApplicationInfo ( ) .loadLabel ( getPackageManager ( ) ) instead of getString ( mApplication.getApplicationInfo ( ) .labelRes ) for retrieving app name .
Fixed GLMModel.java to properly check for too-large coefficients in toJavaCheckTooBig ( ) method .
Added autoencoder support to DeepLearningModel .
Added auxiliary field and methods to LocalActor
Added method &apos; shouldShowModuleName &apos; to PsiDirectoryNode and updated implementation of updateImpl to show module name if it is not equal to directory name
Removed unnecessary constructor from FeedForwardToCnnPreProcessor .
Added try-with-resources block for InputStream and InputStreamReader in DefaultServlet.java to close resources properly .
Added null check for link creation in OPDSLinkXMLReader
Added Windows-specific code in PolicyMonitorTest.java
Added validation for label template in PipelineConfig
Remove unused method parameter &apos; restrictToChildOrNull &apos; in ResolveContext.resolve ( ) and simplify the code .
Added PArray class to Path interface
Fixed issue with language detection in FindManagerImpl
Fixed race condition in ExternalSystemAutoImporter by checking task state before refreshing project
Added check for attribution identifiers in GraphRequest
Added support for managed performance counters in Camel
Modified RenderScript.java to use a local variable for the script ID instead of hardcoding it .
Added try-catch block to handle null reference in getSelectedTemplate ( ) method ( # 3826 )
Added shortcut key for opening in browser action
Update dx command prefix to use / usr / bin / dx instead of / usr / bin / dx &amp;
Added unit test mode check in CompletionProgressIndicator .
Added domain lock and magic create domain to hosts list in XmppConnectionService .
Replace getDefaultConfig ( ) with getIniTemplate ( ) in ModuleGraphWriter .
Added support for displaying asset bundle entries in the tree cell renderer .
Added LuceneTestCase import and await fix for bug URL http : / / build-us-00.elastic.co / job / es _ core _ master _ windows-2012-r2 / 2074 / testReport / ( boaz on it )
Added &apos; getDescriptor ( ) &apos; method to &apos; InternalTaskProgressEvent &apos; interface .
Added keyboard shortcut for interrupting R execution in TextEditingTarget
Updated JBScrollPane to use ViewportBorder instead of creating a new JScrollPane with an empty border .
Added field &apos; myOriginalFile &apos; to OffsetTranslator and updated isUpToDate ( ) method to check for file validity .
Added synchronized block to quitUpdateThread method in DrawHandler.java to ensure thread safety .
Update StatusBarPolicy to use VIBRATE _ TYPE _ RINGER instead of shouldVibrate ( ) for determining the icon ID .
Added BouncyCastleProvider to security providers
Fixed bug in Http2UpgradeHandler.java : incorrect window size decrement
Added import for JavaHighlightUtil and updated TypeMayBeWeakenedInspection to ignore serialization-related methods .
Animating wallpaper crop update
Unlock animation : skip backdrop fade-out and use a null drawable instead
TextInputListener is now called on the rendering thread .
Added filters field and getter method to Emma class , and updated EmmaMojo to use filters when processing instructions .
Allow overwrite of existing data in Tachyon file system
Removed import of io.netty.handler.codec.http.FullHttpRequest and io.netty.handler.codec.http.FullHttpResponse in ResponseHandler.java. Removed implementation of ContentRequestExtractor.java and updated to implement RequestExtractor interface .
Reset progressive start to zero in SmoothProgressDrawable
Added flag show wallpaper in PhoneWindow.java
Fix for PY-1234 : Unload stubs correctly after tree element removal
Changed overscroll distance from 2 to 0
Replace BroadcasterCache with BroadcasterCacheInspector in BroadcasterCacheInspectorServiceProcessor .
Update ProgressIndicator and ContentView to handle redirected URLs and improve progress tracking .
Added serialVersionUID field to ParameterException class .
Reordered parameters in FileSystemMaster.getFileInfo and added missing permission check for create file
Fixed issue with move to top / bottom menu items not being visible when they should be .
Added PEEK _ HEIGHT _ AUTO constant to BottomSheetBehavior
Fixed caching directed graph walker creation for DefaultLenientConfiguration .
Modified LogFormatter to show all exceptions or caught exceptions when showAll or showCaughtException is true .
Added requirement for packet extensions to be included in IQ sub-element XML section
Added &apos; result &apos; field to ActionAnnotation class .
Fixed issue with FileProcessingCompiler.ProcessingItem.getFile ( ) returning null
Removed unnecessary dependency on SYSTEM module
Added lookupKey field to CallerInfo and updated query to include it .
Added flag to start indexing tours after download and scale images to fit screen size .
Fix syntax errors in BuildFileASTTest.java
Removed unnecessary imports and added CountDownLatch import .
Removed unused exchange cache and added a todo for a thread id / tree name based exchange cache in PersistitTransaction .
Allow requesting entity to probe presence of user .
Added check for animation ending when starting a new one
Removed unnecessary border in PushLog .
Update impassible roads menu controller to use dark theme icons .
Added property values to TestingConnectorSession constructor .
Added setter methods for server and jdbcDriver properties in AbstractMetaDataExportMojo .
Changed indentation calculation for soft wraps .
Renamed TestEmployeeTable.class.getCanonicalName ( ) to TestEmployeeTable.class.getSimpleName ( )
Added field &apos; mDate &apos; to class AppLog.LogEntry and updated constructor to include it .
Added import statement for XmlStringUtil and modified descriptionTemplate assignment to use stripHtml method .
Added method getDefaultBlockSize ( ) to FileSystem and DistributedFileSystem classes .
Update AclExampleTest.java to use &apos; Access Control Lists &apos; instead of &apos; ACLs &apos; for better clarity and consistency .
Removed unnecessary cast to HttpURLConnection
Moved FrodoBridge.makeIntent ( ) from public to private and added a new static method startActivity ( Intent , Context ) for starting activities .
Added @ Override annotation to WebService class .
Added test case for StringUtils.replace ( ) method .
Added support for specifying key and value serializers and deserializers in KafkaEndpoint .
Added support for PSK ciphers in CipherSuiteConverter .
Added functionality to hide content view when expanded is false
Added delegation to parent role in ORole.allow ( ) method .
Replace YAML parser with Jackson &apos;s YAML parser
Fixed bug in GraphSONWriterTestSuite where ignoresSuppliedIds was not being checked properly .
Added double tap gesture detection to MessageView
Removed unnecessary imports : SimpleDateFormat , Locale , TimeZone
Added ability to remove libraries from project structure
Added null check for ApplicationManager.getApplication ( ) before getting project data
Set whether methods are handled ( default true ) in GenericExporter .
Temporarily allow user activity during proximity sensor negative events
Removed unused imports and simplified stubbing with subtypes test .
Removed unnecessary import statement for &apos; RemotePropagatable &apos; .
Updated FileHistoryPanelImpl to use custom diff handler and added support for tree view
Added protection for buildInputMethodListLocked and setInputMethodLocked methods using Binder.clearCallingIdentity ( ) and Binder.restoreCallingIdentity ( )
Added support for Spark 1.3 in SparkSqlInterpreter
Added &apos; hasBogusName &apos; method to OpenStreetMapGraphBuilderImpl and updated &apos; setBogusName &apos; to use it .
Changed import from GridMarkerClusterer to RadiusMarkerClusterer in CellTowerGridMarkerClusterer.java
Added Android Layout and Strings files to ImportantFiles .
Remove unsupported cancellation of ListenableFuture in Service interface
